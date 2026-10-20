# Anchor hot cache: ship-fix review (follow-up to your integration review)

You reviewed the integration (e9d65055e3); your disposition table is below.
This commit (dde5b29b6d) applies the fixes for shipping. Do not run commands
(your sandbox cannot run here); review the diff as given.

Changes vs the reviewed version:
- VLLM_ANCHOR_HOT_CACHE is now bool: on = the old "free" (the anchors' HBM
  goes to hot experts). VLLM_ANCHOR_HOT_CACHE_SAME_PLAN (benchmarking only)
  keeps the charge.
- Refuse without DCP or with kv_transfer_config.
- Hot caches built in bind_kv_cache (both runners call it after allocating
  KV, before warmup/capture); extension compiled there.
- Reserved-row write moved from do_kv_cache_update (main) to start_mqa on the
  side stream, with a per-anchor reserved_map (was one shared map built at
  the first anchor). k_c_normed/k_pe/slot slice record_stream(side).
- Inline (non-forked) map path asserts FORK is off (its reserved rows are
  written in kv_update only then).
- Runner block zeroing (KVBlockZeroer.zero_block_ids) and copy-on-write
  (copy_kv_cache_blocks_inplace dst) invalidate those blocks' slots
  (block*64 + 0..63) in every hot cache.
- Guards: map asserts numel <= max_entries and T <= 64; a Python flag
  expect_map is set by a kv_update that skips invalidation and cleared by
  map; kv_update raises if it is still set (a mapped step that did not map),
  map asserts it is set.
- SkipKVStager.stage records its stream on indices and new_slots.

Tests: login-GPU module test (build from a fake forward context, 300 steps x
2 anchors with prefill invalidations, hooked CoW, unhooked CoW control):
decode 0 mismatches, hooked CoW 0, unhooked control 350 (test is sensitive),
both guards trip. Booster in-situ CHECK=1 and served A/B vs prod are running.

Questions:
1. Is the side-stream reserved write correctly ordered? side.wait_stream(main)
   after the main-stream Grace KV write; reserved write then map on side;
   FlashMLA on main waits on hot.ready. Next step's side work vs this step's
   main FlashMLA read of the buffer: the side waits main at each fork.
2. The expect_map flag is Python state evaluated only at capture time under
   FULL CUDA graphs and at each eager step; is there a path (piecewise
   graphs, a dummy run, profile run, a step whose attention is skipped,
   MTP drafter runs that share the target's metadata) where it misfires or
   is left set?
3. Are zeroing and CoW the complete set of non-attention KV writers in this
   config (no connector, no CPU offload)? Anything in the drafter, the
   prefix-cache path, or DCP prefill (forward_mha writes?) missed?
4. build() keyed by data_ptr: any rebind path (sleep mode, elastic, KV
   reinit) where stale caches survive?
5. Anything else blocking ship.

## Your previous disposition
# Astra integration review (ahc-b1 = e9d65055e3): disposition

Verdict: the stream ordering (KV write → side-stream map → main FlashMLA →
next map) is sound; no demonstrated race. Points:

| finding | disposition |
|---|---|
| `record_stream(main)` on side-stream conversion outputs is right; also record the stager stream's read of `indices` (now protected only transitively) | Agreed. Not a bug in this config. Add `indices.record_stream(self.stream)` in `SkipKVStager.stage` before shipping (not in the frozen A/B tree) |
| P1? External KV writers (connectors, swap/offload restore, block copy) bypass `do_kv_cache_update` | None are enabled here (no kv_transfer_config, no CPU offload or swap in v1; the MTP drafter writes only layer 78, which has no hot cache). To do: refuse `VLLM_ANCHOR_HOT_CACHE` when a KV connector is configured |
| P1? Map and update predicates are not literally the same (`md[0]`, `fp8_use_mixed_batch`, prefix vs full slot_mapping) | Under DCP the mixed-batch path is the only one, `md[0]` is the target's metadata under speculative decoding, and padding is -1. To do: per-call checks (`numel <= max_entries`, `t <= 64`) and assert that an update which skips invalidation is followed by a map |
| Invalidation leaves `status` untouched | Checked against the kernel: a row with `slot_of = -1` is skipped on eviction (`old < 0`), and its stale stamp only delays its reuse by one sweep |
| FlashMLA cache size / non-scheduled path | The served path is the scheduled one (DCP4 + VLLM_DCP_DECODE_SCHED). Check mode asserts it. The in-situ check below covers it bitwise, including the reserved rows that are re-quantized by `concat_and_cache_mla` |
| Need a Grace-vs-Grace determinism control | Not needed for this result: nondeterminism could only cause false mismatches, never hide one, and we saw 0 mismatches |
| Host counter: non-atomic adds, unsynchronized snapshot | The adds are serialized on main. Treat the log line as telemetry; the final reading was stable over 30 s |
| `"same"` needs an assertion that the retained charge covers the hot caches | The hot caches are 0.24 GiB vs a 2.57 GiB retained charge at the 800K pool. To do: assert |
| Initialize before capture; scope the singleton; recompute the reserved map per anchor | It works as is (extension and tables are built at the first forward, before capture). Before shipping: move the build to post-KV-allocation init |

In-situ check (hold 2412782, `launch_check.sh`, dec4 at c=4 config,
`VLLM_ANCHOR_HOT_CACHE=same`, `VLLM_ANCHOR_HOT_CACHE_CHECK=1`): ~11,357
mapped anchor calls per rank (~540 verify steps × 21 anchors) at ~5K and
~50K context, **0 mismatching output/LSE elements** against FlashMLA on the
Grace cache. Planner: 14,284 hot / 4,916 cold experts per rank, the same as
prod. Observed free HBM after startup 5.14 GiB vs prod 2.84–2.91 GiB
(+2.3 GiB, as planned).

## Diff (dde5b29b6d vs e9d65055e3 + 08fd07c66c auto size)
```diff
diff --git a/vllm/envs.py b/vllm/envs.py
index 14101c5ec5..cd602a8e30 100644
--- a/vllm/envs.py
+++ b/vllm/envs.py
@@ -281,8 +281,9 @@ if TYPE_CHECKING:
     VLLM_DCP_SPARSE_PREFILL: bool = True
     VLLM_DCP_ONE_SHOT_COLLECTIVES: bool = False
     VLLM_DCP_SPARSE_DECODE_WIDTH: int = 0
-    VLLM_ANCHOR_HOT_CACHE: str = ""
-    VLLM_ANCHOR_HOT_CACHE_ROWS: int = 16384
+    VLLM_ANCHOR_HOT_CACHE: bool = False
+    VLLM_ANCHOR_HOT_CACHE_SAME_PLAN: bool = False
+    VLLM_ANCHOR_HOT_CACHE_ROWS: int = 0
     VLLM_ANCHOR_HOT_CACHE_FORK: bool = True
     VLLM_ANCHOR_HOT_CACHE_CHECK: bool = False
     VLLM_DCP_DECODE_SCHED: bool = True
@@ -2021,13 +2022,15 @@ environment_variables: dict[str, Callable[[], Any]] = {
         os.getenv("VLLM_DCP_SPARSE_DECODE_WIDTH", "0")
     ),
     # Anchor layers' main MLA cache on Grace behind a per-layer HBM hot cache
-    # (skip_host_uva only): "same" keeps the planner's HBM charge for it,
-    # "free" gives that HBM to the planner; empty disables.
-    "VLLM_ANCHOR_HOT_CACHE": env_with_choices(
-        "VLLM_ANCHOR_HOT_CACHE", "", ["", "same", "free"]
+    # (skip_host_uva with DCP); the planner gives the freed HBM to hot experts
+    "VLLM_ANCHOR_HOT_CACHE": lambda: bool(int(os.getenv("VLLM_ANCHOR_HOT_CACHE", "0"))),
+    # benchmarking only: keep the anchors' HBM charge (the hot set without it)
+    "VLLM_ANCHOR_HOT_CACHE_SAME_PLAN": lambda: bool(
+        int(os.getenv("VLLM_ANCHOR_HOT_CACHE_SAME_PLAN", "0"))
     ),
+    # hot rows per anchor; 0 sizes them from the decode step (anchor_hot_cache)
     "VLLM_ANCHOR_HOT_CACHE_ROWS": lambda: int(
-        os.getenv("VLLM_ANCHOR_HOT_CACHE_ROWS", "16384")
+        os.getenv("VLLM_ANCHOR_HOT_CACHE_ROWS", "0")
     ),
     # map the anchor's top-k on a side stream from the start of attention
     "VLLM_ANCHOR_HOT_CACHE_FORK": lambda: bool(
@@ -2437,6 +2440,7 @@ def compile_factors() -> dict[str, object]:
         "VLLM_DCP_SPARSE_DECODE_WIDTH",
         "VLLM_DCP_DECODE_SCHED",
         "VLLM_ANCHOR_HOT_CACHE",
+        "VLLM_ANCHOR_HOT_CACHE_SAME_PLAN",
         "VLLM_ANCHOR_HOT_CACHE_ROWS",
         "VLLM_ANCHOR_HOT_CACHE_FORK",
         "VLLM_ANCHOR_HOT_CACHE_CHECK",
diff --git a/vllm/model_executor/layers/attention/mla_attention.py b/vllm/model_executor/layers/attention/mla_attention.py
index 38e8a29e07..33b07b280e 100644
--- a/vllm/model_executor/layers/attention/mla_attention.py
+++ b/vllm/model_executor/layers/attention/mla_attention.py
@@ -774,7 +774,14 @@ class MLAAttention(nn.Module, AttentionLayerBase):
         if num_mqa_tokens > 0:
             start_mqa = getattr(self.impl, "start_mqa", None)
             if start_mqa is not None:
-                start_mqa(kv_cache, attn_metadata, num_mqa_tokens)
+                start_mqa(
+                    kv_cache,
+                    attn_metadata,
+                    num_mqa_tokens,
+                    k_c_normed,
+                    k_pe,
+                    self._k_scale,
+                )
             mqa_q = q[:num_mqa_tokens]
             mqa_output_slice = output[:num_mqa_tokens]
 
diff --git a/vllm/model_executor/model_loader/tiered_moe_kv.py b/vllm/model_executor/model_loader/tiered_moe_kv.py
index a258dd9f11..0526d65f9e 100644
--- a/vllm/model_executor/model_loader/tiered_moe_kv.py
+++ b/vllm/model_executor/model_loader/tiered_moe_kv.py
@@ -88,6 +88,11 @@ def anchor_hot_cache_layers(vllm_config: "VllmConfig") -> tuple[int, ...]:
         return ()
     if vllm_config.tiered_moe_config.mla_cache_tier != "skip_host_uva":
         raise ValueError("VLLM_ANCHOR_HOT_CACHE requires skip_host_uva")
+    if vllm_config.parallel_config.decode_context_parallel_size <= 1:
+        raise ValueError("VLLM_ANCHOR_HOT_CACHE requires DCP")
+    if vllm_config.kv_transfer_config is not None:
+        # a connector writes KV blocks the hot cache does not see
+        raise ValueError("VLLM_ANCHOR_HOT_CACHE does not support KV transfer")
     return glm_full_indexer_layers(num_hidden_layers=78)
 
 
diff --git a/vllm/model_executor/model_loader/tiered_moe_physical.py b/vllm/model_executor/model_loader/tiered_moe_physical.py
index c72d36739e..2f5f4c3000 100644
--- a/vllm/model_executor/model_loader/tiered_moe_physical.py
+++ b/vllm/model_executor/model_loader/tiered_moe_physical.py
@@ -321,22 +321,34 @@ def plan_tiered_moe_scenario(
     )
     main_hbm_bytes, host_bytes = kv_plan.main_hbm_bytes, kv_plan.host_bytes
     if envs.VLLM_ANCHOR_HOT_CACHE and cache_tier == "skip_host_uva":
-        from vllm.v1.attention.backends.mla import anchor_hot_cache, skip_kv_stage
+        from vllm.v1.attention.backends.mla import anchor_hot_cache
 
-        # the target's anchors move to Grace; "same" keeps their HBM charge
-        # (the hot set is skip_host_uva's), "free" charges their hot caches
+        # the target's anchors move to Grace and their HBM goes to the hot
+        # caches and hot experts (SAME_PLAN, for benchmarking: kept charged)
         moved = len(kv_plan.indexer_layer_ids) - num_mtp_layers
         per_layer = kv_plan.main_cache_bytes // kv_plan.main_layer_count
+        num_slots = per_layer // kv_plan.main_page_bytes_per_layer * kv_plan.block_size
+        hot_bytes = anchor_hot_cache.table_bytes(num_slots)
+        if hot_bytes >= per_layer:
+            raise ValueError(
+                f"Anchor hot cache ({hot_bytes} B per layer) is not smaller than "
+                f"the anchor's cache ({per_layer} B): unset VLLM_ANCHOR_HOT_CACHE"
+            )
         host_bytes += moved * per_layer
-        if envs.VLLM_ANCHOR_HOT_CACHE == "free":
-            queries, columns = skip_kv_stage.limits()
+        if not envs.VLLM_ANCHOR_HOT_CACHE_SAME_PLAN:
             main_hbm_bytes -= moved * per_layer
-            fixed_hbm_allocations["anchor_hot_cache"] = moved * (
-                anchor_hot_cache.table_bytes(
-                    per_layer // kv_plan.main_page_bytes_per_layer * kv_plan.block_size,
-                    queries * columns,
-                )
-            )
+            fixed_hbm_allocations["anchor_hot_cache"] = moved * hot_bytes
+        logger.info(
+            "Anchor hot cache: %d rows per layer (%d entries per step), "
+            "%.3f GiB over %d layers%s",
+            anchor_hot_cache.rows(),
+            anchor_hot_cache.max_entries(),
+            moved * hot_bytes / 1024**3,
+            moved,
+            " (anchors' HBM charge kept)"
+            if envs.VLLM_ANCHOR_HOT_CACHE_SAME_PLAN
+            else "",
+        )
     if main_hbm_bytes:
         fixed_hbm_allocations["main_mla_cache"] = main_hbm_bytes
     if host_bytes:
diff --git a/vllm/v1/attention/backends/mla/anchor_hot_cache.py b/vllm/v1/attention/backends/mla/anchor_hot_cache.py
index c2584e59d2..b9d62fa337 100644
--- a/vllm/v1/attention/backends/mla/anchor_hot_cache.py
+++ b/vllm/v1/attention/backends/mla/anchor_hot_cache.py
@@ -2,22 +2,22 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """HBM hot cache for anchor layers whose main MLA cache lives on Grace.
 
-With ``VLLM_ANCHOR_HOT_CACHE`` (``same`` or ``free``) under ``skip_host_uva``
+Under ``skip_host_uva`` with DCP and ``VLLM_ANCHOR_HOT_CACHE=1``
 the target's anchor layers (those with their own DSA indexer) keep their
 fp8_ds_mla main cache in pinned Grace memory too; the grafted MTP layer stays
-in HBM. Each such layer keeps ``VLLM_ANCHOR_HOT_CACHE_ROWS`` rows of it in an
-HBM buffer, plus RESERVED_ROWS rows for this step's tokens.
-
-In a decode step the anchor's converted top-k (this rank's slots) goes
-through one cooperative kernel (anchor_hot.cu): a hit keeps its row, a miss
-takes a CLOCK victim and is copied from Grace, and this step's token j maps to
-row ``rows + j``, which the layer's KV write fills. FlashMLA then reads the
-buffer with the mapped indices. Entries keep position, multiplicity and -1, so
-it reads the same bytes in the same order. Any other step reads Grace
-directly and drops the slots it writes from the buffer.
-
-``same`` keeps the planner's HBM charge for these caches, so the hot expert
-set is skip_host_uva's; ``free`` hands the freed HBM to the planner.
+in HBM. Each such layer keeps ``rows()`` rows of it in an HBM buffer, plus
+RESERVED_ROWS rows for this step's tokens. The planner gives the freed HBM to
+hot experts.
+
+In a decode step the anchor forks a side stream at the start of attention
+(start_mqa): this step's tokens go to the reserved rows, then the converted
+top-k (this rank's slots) goes through one cooperative kernel (anchor_hot.cu):
+a hit keeps its row, a miss takes a CLOCK victim and is copied from Grace, and
+token j maps to row ``rows + j``. FlashMLA then reads the buffer with the
+mapped indices. Entries keep position, multiplicity and -1, so it reads the
+same bytes in the same order. Any other step reads Grace directly and drops
+the slots it writes from the buffer; so do the runner's block zeroing and
+copy-on-write block copies.
 """
 
 from functools import cache
@@ -35,13 +35,10 @@ _HERE = Path(__file__).resolve().parent
 RESERVED_ROWS = 64  # one per query token of a step (anchor_hot.cu: t <= 64)
 ROW_BYTES = 656
 _CTL = 32
+MIN_ROWS = 4096  # anchor_hot.cu's sweep window (THREADS x SWEEP_V)
 _ANCHORS: frozenset[int] = frozenset()
 
 
-def enabled() -> bool:
-    return envs.VLLM_ANCHOR_HOT_CACHE in ("same", "free")
-
-
 def configure(vllm_config) -> None:
     """Record the hot anchors while the config is set (layer init)."""
     global _ANCHORS
@@ -52,13 +49,40 @@ def configure(vllm_config) -> None:
     _ANCHORS = frozenset(anchor_hot_cache_layers(vllm_config))
 
 
-def table_bytes(num_slots: int, max_entries: int) -> int:
+def max_entries() -> int:
+    """The most index entries one decode step maps per anchor: the staging
+    limits (max_num_seqs x (1 + speculative tokens), at most 64, times the
+    DCP decode width or the top-k)."""
+    from vllm.v1.attention.backends.mla import skip_kv_stage
+
+    queries, columns = skip_kv_stage.limits()
+    return queries * columns
+
+
+def rows() -> int:
+    """Hot rows per anchor: VLLM_ANCHOR_HOT_CACHE_ROWS if set, else 4/3 of
+    max_entries() (16384 at c=4 MTP3 under DCP4, the replayed size; 8192 at
+    c=1 MTP7, 65536 at c=16 MTP3). The kernel needs at least max_entries()
+    rows (every claim must find a victim) and its 4096-row sweep window, so
+    every decode step maps; only non-decode steps read Grace."""
+    entries = max_entries()
+    n = envs.VLLM_ANCHOR_HOT_CACHE_ROWS or -(-entries * 4 // 3)
+    n = max(-(-n // 64) * 64, MIN_ROWS)
+    if n < entries:
+        raise ValueError(
+            f"VLLM_ANCHOR_HOT_CACHE_ROWS={n} is below the {entries} entries "
+            "a decode step can map"
+        )
+    return n
+
+
+def table_bytes(num_slots: int) -> int:
     """HBM one layer's hot cache holds: buffer, slot/row maps, miss lists."""
-    rows = envs.VLLM_ANCHOR_HOT_CACHE_ROWS
+    rows_, entries = rows(), max_entries()
     return (
-        (rows + RESERVED_ROWS) * ROW_BYTES
-        + 4 * (num_slots + 2 * rows + _CTL + 3 * max_entries)
-        + 8 * RESERVED_ROWS
+        (rows_ + RESERVED_ROWS) * ROW_BYTES
+        + 4 * (num_slots + 2 * rows_ + _CTL + 3 * entries)
+        + 16 * RESERVED_ROWS
     )
 
 
@@ -93,10 +117,10 @@ def _invalidate_kernel(slots_ptr, n, where_ptr, slot_of_ptr, BLOCK: tl.constexpr
 class AnchorHotCache:
     """One anchor layer's buffer and maps (anchor_hot.cu's State)."""
 
-    def __init__(self, grace_cache: torch.Tensor, max_entries: int):
-        rows = envs.VLLM_ANCHOR_HOT_CACHE_ROWS
+    def __init__(self, grace_cache: torch.Tensor, rows: int, max_entries: int):
         # the kernel's sweep window and victim guarantee (anchor_hot_map)
-        assert rows % 64 == 0 and rows >= 4096 and max_entries <= rows
+        assert rows % 64 == 0 and rows >= MIN_ROWS and max_entries <= rows
+        assert grace_cache.dtype == torch.uint8 and grace_cache.shape[-1] == ROW_BYTES
         device = grace_cache.device
         self.grace = grace_cache
         self.rows = rows
@@ -118,12 +142,48 @@ class AnchorHotCache:
         self.miss = torch.zeros(max_entries, **i32)
         self.miss_row = torch.zeros(max_entries, **i32)
         self.out = torch.full((max_entries,), -1, **i32)
+        self.reserved = rows + torch.arange(
+            RESERVED_ROWS, dtype=torch.int64, device=device
+        )
+        self.reserved_map = torch.full_like(self.reserved, -1)
         self.ready = torch.cuda.Event()
+        # set by a KV write that left its slots to the map, cleared by the map
+        self.expect_map = False
+
+    def write_reserved(
+        self,
+        kv_c_normed: torch.Tensor,
+        k_pe: torch.Tensor,
+        slot_mapping: torch.Tensor,
+        k_scale: torch.Tensor,
+    ) -> None:
+        """This step's tokens into the reserved rows (rows + j for token j,
+        skipped where another DCP rank owns it), as the Grace write packs
+        them."""
+        from vllm import _custom_ops as ops
+
+        t = slot_mapping.numel()
+        assert t <= RESERVED_ROWS
+        self.reserved_map[:t] = torch.where(slot_mapping >= 0, self.reserved[:t], -1)
+        ops.concat_and_cache_mla(
+            kv_c_normed,
+            k_pe.squeeze(1),
+            self.buffer,
+            self.reserved_map[:t],
+            kv_cache_dtype="fp8_ds_mla",
+            scale=k_scale,
+        )
 
     def map(self, indices: torch.Tensor, new_slots: torch.Tensor) -> torch.Tensor:
         """Buffer rows for ``indices`` [T, W] (this rank's slots or -1);
         ``new_slots``: this step's slot mapping [T]."""
         n = indices.numel()
+        assert n <= self.max_entries and new_slots.numel() <= RESERVED_ROWS, (
+            f"anchor hot cache sized for {self.max_entries} entries, got "
+            f"{tuple(indices.shape)}"
+        )
+        assert self.expect_map, "anchor hot cache map without its KV write"
+        self.expect_map = False
         out = self.out[:n]
         _extension().anchor_hot_map(
             indices.reshape(-1),
@@ -140,11 +200,11 @@ class AnchorHotCache:
         )
         return out.view(indices.shape)
 
-    def invalidate(self, slot_mapping: torch.Tensor) -> None:
-        n = slot_mapping.numel()
+    def invalidate(self, slots: torch.Tensor) -> None:
+        n = slots.numel()
         if n:
             _invalidate_kernel[(triton.cdiv(n, 1024),)](
-                slot_mapping, n, self.where, self.slot_of, BLOCK=1024
+                slots, n, self.where, self.slot_of, BLOCK=1024
             )
 
 
@@ -154,23 +214,18 @@ class AnchorHotCaches:
     def __init__(self, device: torch.device, layers: dict[int, Any]):
         from vllm.v1.attention.backends.mla import skip_kv_stage
 
-        queries, columns = skip_kv_stage.limits()
-        self.max_queries = queries
+        self.max_queries = skip_kv_stage.limits()[0]
+        n, entries = rows(), max_entries()
         self.caches: dict[int, AnchorHotCache] = {}
         self.names: dict[int, str] = {}
         for index in sorted(layers):
             name, layer = layers[index]
             if index in _ANCHORS and not layer.impl._reuses_topk:
                 ptr = layer.kv_cache.data_ptr()
-                self.caches[ptr] = AnchorHotCache(layer.kv_cache, queries * columns)
+                self.caches[ptr] = AnchorHotCache(layer.kv_cache, n, entries)
                 self.names[ptr] = name
         self.stream = torch.cuda.Stream(device)
-        self.first = next(iter(self.caches), None)
-        rows = envs.VLLM_ANCHOR_HOT_CACHE_ROWS
-        self._reserved = rows + torch.arange(
-            RESERVED_ROWS, dtype=torch.int64, device=device
-        )
-        self.reserved_map = torch.full_like(self._reserved, -1)
+        self.block_offsets = torch.arange(64, dtype=torch.int64, device=device)
 
     def decodes(self, attn_metadata: Any) -> bool:
         """Whether the step maps through the hot cache (else Grace is read
@@ -178,7 +233,7 @@ class AnchorHotCaches:
         return (
             attn_metadata is not None
             and attn_metadata.num_prefills == 0
-            and attn_metadata.num_actual_tokens <= self.max_queries
+            and 0 < attn_metadata.num_actual_tokens <= self.max_queries
         )
 
     def kv_update(
@@ -189,57 +244,64 @@ class AnchorHotCaches:
         slot_mapping: torch.Tensor,
         k_scale: torch.Tensor,
     ) -> None:
-        """Anchor's KV write (after Grace's): in a mapped step this step's
-        tokens also go to its reserved rows (rows + j for token j, skipped
-        where another DCP rank owns it); in any other step the written slots
-        leave the buffer."""
+        """Anchor's KV write (after Grace's). A mapped step leaves its slots
+        to the map, which writes the reserved rows (inline: written here);
+        any other step drops the written slots from the buffer."""
         ptr = grace_cache.data_ptr()
-        hot = self.caches.get(ptr)
-        if hot is None:
-            return
-        md = get_forward_context().attn_metadata
+        hot = self.caches[ptr]
+        # the attention layer's own metadata (MLAAttention.forward's lookup)
+        md: Any = get_forward_context().attn_metadata
         if isinstance(md, list):
             md = md[0]
         if isinstance(md, dict):
             md = md.get(self.names[ptr])
+        if hot.expect_map:
+            raise RuntimeError("anchor hot cache: a mapped step did not map")
         if not self.decodes(md):
             hot.invalidate(slot_mapping)
             return
-        t = slot_mapping.numel()
-        assert t <= RESERVED_ROWS
-        from vllm import _custom_ops as ops
+        hot.expect_map = True
+        if not envs.VLLM_ANCHOR_HOT_CACHE_FORK:
+            hot.write_reserved(kv_c_normed, k_pe, slot_mapping, k_scale)
 
-        if ptr == self.first:  # the same for every anchor
-            self.reserved_map[:t] = torch.where(
-                slot_mapping >= 0, self._reserved[:t], -1
-            )
-        ops.concat_and_cache_mla(
-            kv_c_normed,
-            k_pe.squeeze(1),
-            hot.buffer,
-            self.reserved_map[:t],
-            kv_cache_dtype="fp8_ds_mla",
-            scale=k_scale,
-        )
+    def invalidate_blocks(self, block_ids: torch.Tensor) -> None:
+        """Drop whole blocks (runner zeroing, copy-on-write destinations)."""
+        if not self.caches or block_ids.numel() == 0:
+            return
+        slots = (block_ids.to(torch.int64)[:, None] * 64 + self.block_offsets).view(-1)
+        for hot in self.caches.values():
+            hot.invalidate(slots)
 
 
 _CACHES: AnchorHotCaches | None = None
 
 
-def get(grace_cache: torch.Tensor) -> AnchorHotCache | None:
-    """The layer's hot cache, else None (not a hot anchor, or caches unbound).
-    The rank's caches are built from the bound attention layers on first use."""
+def build(forward_context: dict[str, Any]) -> None:
+    """Build the rank's hot caches once the KV caches are bound (before any
+    warmup or graph capture), replacing any earlier set."""
     global _CACHES
-    if _CACHES is None:
-        if grace_cache.numel() == 0:
-            return None
-        layers = {
-            extract_layer_index(name): (name, layer)
-            for name, layer in get_forward_context().no_compile_layers.items()
-            if hasattr(getattr(layer, "impl", None), "_reuses_topk")
-        }
-        _CACHES = AnchorHotCaches(grace_cache.device, layers)
-    return _CACHES.caches.get(grace_cache.data_ptr())
+    _CACHES = None
+    if not _ANCHORS:
+        return
+    layers = {
+        extract_layer_index(name): (name, layer)
+        for name, layer in forward_context.items()
+        if hasattr(getattr(layer, "impl", None), "_reuses_topk")
+    }
+    first = next(iter(layers.values()))[1].kv_cache
+    _CACHES = AnchorHotCaches(first.device, layers)
+    assert len(_CACHES.caches) == len(_ANCHORS), "anchor hot cache layers missing"
+    _extension()
+
+
+def get(grace_cache: torch.Tensor) -> AnchorHotCache | None:
+    """The layer's hot cache, else None (not a hot anchor)."""
+    return None if _CACHES is None else _CACHES.caches.get(grace_cache.data_ptr())
+
+
+def invalidate_blocks(block_ids: torch.Tensor) -> None:
+    if _CACHES is not None:
+        _CACHES.invalidate_blocks(block_ids)
 
 
 class _Check:
diff --git a/vllm/v1/attention/backends/mla/flashmla_sparse.py b/vllm/v1/attention/backends/mla/flashmla_sparse.py
index 84b09a99c4..5a91d69a0e 100644
--- a/vllm/v1/attention/backends/mla/flashmla_sparse.py
+++ b/vllm/v1/attention/backends/mla/flashmla_sparse.py
@@ -697,7 +697,7 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
         # anchor's main cache on Grace behind an HBM hot cache (the grafted MTP
         # layer has none: anchor_hot_cache.get returns None for it)
         self._hot_anchor = (
-            self._skip_kv and not self._reuses_topk and anchor_hot_cache.enabled()
+            self._skip_kv and not self._reuses_topk and envs.VLLM_ANCHOR_HOT_CACHE
         )
         if self._hot_anchor:
             anchor_hot_cache.configure(vllm_config)
@@ -917,6 +917,8 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
             )
             hot = self._hot_cache(kv_c_and_k_pe_cache, attn_metadata)
             if hot is not None:
+                # forked mode writes the reserved rows in start_mqa
+                assert not envs.VLLM_ANCHOR_HOT_CACHE_FORK
                 hot_rows = hot.map(
                     topk_indices, attn_metadata.slot_mapping[: q.shape[0]]
                 )
@@ -1097,10 +1099,14 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
         kv_cache: torch.Tensor,
         attn_metadata: FlashMLASparseMetadata,
         num_tokens: int,
+        kv_c_normed: torch.Tensor,
+        k_pe: torch.Tensor,
+        k_scale: torch.Tensor,
     ) -> None:
         """Start of the decode attention, before the query absorb and its
-        all-gather: convert and map this anchor's top-k on the hot cache's
-        stream, so the copy from Grace overlaps them."""
+        all-gather: on the hot cache's stream, write this step's tokens to
+        the reserved rows, then convert and map this anchor's top-k, so the
+        copy from Grace overlaps them."""
         if not envs.VLLM_ANCHOR_HOT_CACHE_FORK:
             return
         hot = self._hot_cache(kv_cache, attn_metadata)
@@ -1110,18 +1116,24 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
         main = torch.cuda.current_stream()
         side = anchor_hot_cache.stream()
         side.wait_stream(main)
+        new_slots = attn_metadata.slot_mapping[:num_tokens]
         with torch.cuda.stream(side):
+            hot.write_reserved(
+                kv_c_normed[:num_tokens], k_pe[:num_tokens], new_slots, k_scale
+            )
             indices, valid_counts, sched = self._converted_indices(
                 self.topk_indices_buffer[:num_tokens],
                 attn_metadata,
                 self._dcp_decode_sched(self.num_heads * self.dcp_world_size),
             )
-            rows = hot.map(indices, attn_metadata.slot_mapping[:num_tokens])
+            rows = hot.map(indices, new_slots)
             hot.ready.record(side)
         # allocated on the side stream, read on main
         for t in (indices, valid_counts, *(sched or ())):
             if t is not None:
                 t.record_stream(main)
+        for t in (kv_c_normed, k_pe, new_slots):
+            t.record_stream(side)
         self._hot_pending = (indices, valid_counts, sched, rows, hot.ready)
 
     def _dcp_converted_indices(
diff --git a/vllm/v1/attention/backends/mla/skip_kv_stage.py b/vllm/v1/attention/backends/mla/skip_kv_stage.py
index 9bd27bb4c0..65ecb4f8c2 100644
--- a/vllm/v1/attention/backends/mla/skip_kv_stage.py
+++ b/vllm/v1/attention/backends/mla/skip_kv_stage.py
@@ -284,6 +284,8 @@ class SkipKVStager:
                     0, (self.count // 256).long(), self.count.new_ones(1).float()
                 )
             ready.record(self.stream)
+        indices.record_stream(self.stream)
+        new.record_stream(self.stream)
         self._ready = ready
 
     def staged(
diff --git a/vllm/v1/worker/utils.py b/vllm/v1/worker/utils.py
index 9e60d2ae7f..ad906c4d24 100644
--- a/vllm/v1/worker/utils.py
+++ b/vllm/v1/worker/utils.py
@@ -10,6 +10,7 @@ from typing import Any
 import numpy as np
 import torch
 
+import vllm.envs as envs
 from vllm.config import CacheConfig, VllmConfig
 from vllm.logger import init_logger
 from vllm.model_executor.layers.attention import Attention
@@ -237,6 +238,15 @@ class KVBlockZeroer:
             PAGE_SIZE_EL=page_size_el,
             BLOCK_SIZE=blk_size,
         )
+        _anchor_hot_cache_drop(idx)
+
+
+def _anchor_hot_cache_drop(block_ids: torch.Tensor) -> None:
+    """Blocks written outside the attention layers' KV update."""
+    if envs.VLLM_ANCHOR_HOT_CACHE:
+        from vllm.v1.attention.backends.mla import anchor_hot_cache
+
+        anchor_hot_cache.invalidate_blocks(block_ids)
 
 
 @dataclass
@@ -537,6 +547,11 @@ def bind_kv_cache(
     for layer_name, kv_cache in kv_caches.items():
         forward_context[layer_name].kv_cache = kv_cache
 
+    if envs.VLLM_ANCHOR_HOT_CACHE:
+        from vllm.v1.attention.backends.mla import anchor_hot_cache
+
+        anchor_hot_cache.build(forward_context)
+
 
 def copy_kv_cache_blocks_inplace(
     kv_caches: Iterable[torch.Tensor | list[torch.Tensor]],
@@ -575,6 +590,7 @@ def copy_kv_cache_blocks_inplace(
         assert blocks.numel() % num_blocks == 0
         blocks = blocks.view(num_blocks, -1)
         blocks[dst_indices] = blocks[src_indices]
+    _anchor_hot_cache_drop(dst_indices)
 
 
 def is_residual_scattered_for_sp(
```

## Full anchor_hot_cache.py
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HBM hot cache for anchor layers whose main MLA cache lives on Grace.

Under ``skip_host_uva`` with DCP and ``VLLM_ANCHOR_HOT_CACHE=1``
the target's anchor layers (those with their own DSA indexer) keep their
fp8_ds_mla main cache in pinned Grace memory too; the grafted MTP layer stays
in HBM. Each such layer keeps ``rows()`` rows of it in an HBM buffer, plus
RESERVED_ROWS rows for this step's tokens. The planner gives the freed HBM to
hot experts.

In a decode step the anchor forks a side stream at the start of attention
(start_mqa): this step's tokens go to the reserved rows, then the converted
top-k (this rank's slots) goes through one cooperative kernel (anchor_hot.cu):
a hit keeps its row, a miss takes a CLOCK victim and is copied from Grace, and
token j maps to row ``rows + j``. FlashMLA then reads the buffer with the
mapped indices. Entries keep position, multiplicity and -1, so it reads the
same bytes in the same order. Any other step reads Grace directly and drops
the slots it writes from the buffer; so do the runner's block zeroing and
copy-on-write block copies.
"""

from functools import cache
from pathlib import Path
from typing import Any

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.utils import extract_layer_index
from vllm.triton_utils import tl, triton

_HERE = Path(__file__).resolve().parent
RESERVED_ROWS = 64  # one per query token of a step (anchor_hot.cu: t <= 64)
ROW_BYTES = 656
_CTL = 32
MIN_ROWS = 4096  # anchor_hot.cu's sweep window (THREADS x SWEEP_V)
_ANCHORS: frozenset[int] = frozenset()


def configure(vllm_config) -> None:
    """Record the hot anchors while the config is set (layer init)."""
    global _ANCHORS
    from vllm.model_executor.model_loader.tiered_moe_kv import (
        anchor_hot_cache_layers,
    )

    _ANCHORS = frozenset(anchor_hot_cache_layers(vllm_config))


def max_entries() -> int:
    """The most index entries one decode step maps per anchor: the staging
    limits (max_num_seqs x (1 + speculative tokens), at most 64, times the
    DCP decode width or the top-k)."""
    from vllm.v1.attention.backends.mla import skip_kv_stage

    queries, columns = skip_kv_stage.limits()
    return queries * columns


def rows() -> int:
    """Hot rows per anchor: VLLM_ANCHOR_HOT_CACHE_ROWS if set, else 4/3 of
    max_entries() (16384 at c=4 MTP3 under DCP4, the replayed size; 8192 at
    c=1 MTP7, 65536 at c=16 MTP3). The kernel needs at least max_entries()
    rows (every claim must find a victim) and its 4096-row sweep window, so
    every decode step maps; only non-decode steps read Grace."""
    entries = max_entries()
    n = envs.VLLM_ANCHOR_HOT_CACHE_ROWS or -(-entries * 4 // 3)
    n = max(-(-n // 64) * 64, MIN_ROWS)
    if n < entries:
        raise ValueError(
            f"VLLM_ANCHOR_HOT_CACHE_ROWS={n} is below the {entries} entries "
            "a decode step can map"
        )
    return n


def table_bytes(num_slots: int) -> int:
    """HBM one layer's hot cache holds: buffer, slot/row maps, miss lists."""
    rows_, entries = rows(), max_entries()
    return (
        (rows_ + RESERVED_ROWS) * ROW_BYTES
        + 4 * (num_slots + 2 * rows_ + _CTL + 3 * entries)
        + 16 * RESERVED_ROWS
    )


@cache
def _extension():
    from torch.utils.cpp_extension import load

    name = "vllm_anchor_hot"
    build = Path(envs.VLLM_CACHE_ROOT) / "torch_extensions" / name
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name=name,
        sources=[str(_HERE / "anchor_hot.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=str(build),
        verbose=False,
    )


@triton.jit
def _invalidate_kernel(slots_ptr, n, where_ptr, slot_of_ptr, BLOCK: tl.constexpr):
    """Drop every written slot from the buffer (both maps)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(slots_ptr + offs, mask=offs < n, other=-1)
    live = s >= 0
    s = tl.where(live, s, 0)
    r = tl.load(where_ptr + s, mask=live, other=-1)
    tl.store(slot_of_ptr + tl.where(r >= 0, r, 0), -1, mask=live & (r >= 0))
    tl.store(where_ptr + s, -1, mask=live)


class AnchorHotCache:
    """One anchor layer's buffer and maps (anchor_hot.cu's State)."""

    def __init__(self, grace_cache: torch.Tensor, rows: int, max_entries: int):
        # the kernel's sweep window and victim guarantee (anchor_hot_map)
        assert rows % 64 == 0 and rows >= MIN_ROWS and max_entries <= rows
        assert grace_cache.dtype == torch.uint8 and grace_cache.shape[-1] == ROW_BYTES
        device = grace_cache.device
        self.grace = grace_cache
        self.rows = rows
        self.max_entries = max_entries
        num_slots = grace_cache.numel() // ROW_BYTES
        self.buffer = torch.zeros(
            (rows + RESERVED_ROWS) // 64,
            64,
            ROW_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        i32 = dict(dtype=torch.int32, device=device)
        self.where = torch.full((num_slots,), -1, **i32)
        self.slot_of = torch.full((rows,), -1, **i32)
        self.status = torch.zeros(rows, **i32)
        self.ctl = torch.zeros(_CTL, **i32)
        self.ctl[10:12] = -1
        self.miss = torch.zeros(max_entries, **i32)
        self.miss_row = torch.zeros(max_entries, **i32)
        self.out = torch.full((max_entries,), -1, **i32)
        self.reserved = rows + torch.arange(
            RESERVED_ROWS, dtype=torch.int64, device=device
        )
        self.reserved_map = torch.full_like(self.reserved, -1)
        self.ready = torch.cuda.Event()
        # set by a KV write that left its slots to the map, cleared by the map
        self.expect_map = False

    def write_reserved(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> None:
        """This step's tokens into the reserved rows (rows + j for token j,
        skipped where another DCP rank owns it), as the Grace write packs
        them."""
        from vllm import _custom_ops as ops

        t = slot_mapping.numel()
        assert t <= RESERVED_ROWS
        self.reserved_map[:t] = torch.where(slot_mapping >= 0, self.reserved[:t], -1)
        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            self.buffer,
            self.reserved_map[:t],
            kv_cache_dtype="fp8_ds_mla",
            scale=k_scale,
        )

    def map(self, indices: torch.Tensor, new_slots: torch.Tensor) -> torch.Tensor:
        """Buffer rows for ``indices`` [T, W] (this rank's slots or -1);
        ``new_slots``: this step's slot mapping [T]."""
        n = indices.numel()
        assert n <= self.max_entries and new_slots.numel() <= RESERVED_ROWS, (
            f"anchor hot cache sized for {self.max_entries} entries, got "
            f"{tuple(indices.shape)}"
        )
        assert self.expect_map, "anchor hot cache map without its KV write"
        self.expect_map = False
        out = self.out[:n]
        _extension().anchor_hot_map(
            indices.reshape(-1),
            new_slots,
            out,
            self.grace,
            self.buffer,
            self.where,
            self.slot_of,
            self.status,
            self.ctl,
            self.miss,
            self.miss_row,
        )
        return out.view(indices.shape)

    def invalidate(self, slots: torch.Tensor) -> None:
        n = slots.numel()
        if n:
            _invalidate_kernel[(triton.cdiv(n, 1024),)](
                slots, n, self.where, self.slot_of, BLOCK=1024
            )


class AnchorHotCaches:
    """The rank's hot caches, keyed by the anchors' Grace cache pointers."""

    def __init__(self, device: torch.device, layers: dict[int, Any]):
        from vllm.v1.attention.backends.mla import skip_kv_stage

        self.max_queries = skip_kv_stage.limits()[0]
        n, entries = rows(), max_entries()
        self.caches: dict[int, AnchorHotCache] = {}
        self.names: dict[int, str] = {}
        for index in sorted(layers):
            name, layer = layers[index]
            if index in _ANCHORS and not layer.impl._reuses_topk:
                ptr = layer.kv_cache.data_ptr()
                self.caches[ptr] = AnchorHotCache(layer.kv_cache, n, entries)
                self.names[ptr] = name
        self.stream = torch.cuda.Stream(device)
        self.block_offsets = torch.arange(64, dtype=torch.int64, device=device)

    def decodes(self, attn_metadata: Any) -> bool:
        """Whether the step maps through the hot cache (else Grace is read
        directly and written slots are dropped)."""
        return (
            attn_metadata is not None
            and attn_metadata.num_prefills == 0
            and 0 < attn_metadata.num_actual_tokens <= self.max_queries
        )

    def kv_update(
        self,
        grace_cache: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> None:
        """Anchor's KV write (after Grace's). A mapped step leaves its slots
        to the map, which writes the reserved rows (inline: written here);
        any other step drops the written slots from the buffer."""
        ptr = grace_cache.data_ptr()
        hot = self.caches[ptr]
        # the attention layer's own metadata (MLAAttention.forward's lookup)
        md: Any = get_forward_context().attn_metadata
        if isinstance(md, list):
            md = md[0]
        if isinstance(md, dict):
            md = md.get(self.names[ptr])
        if hot.expect_map:
            raise RuntimeError("anchor hot cache: a mapped step did not map")
        if not self.decodes(md):
            hot.invalidate(slot_mapping)
            return
        hot.expect_map = True
        if not envs.VLLM_ANCHOR_HOT_CACHE_FORK:
            hot.write_reserved(kv_c_normed, k_pe, slot_mapping, k_scale)

    def invalidate_blocks(self, block_ids: torch.Tensor) -> None:
        """Drop whole blocks (runner zeroing, copy-on-write destinations)."""
        if not self.caches or block_ids.numel() == 0:
            return
        slots = (block_ids.to(torch.int64)[:, None] * 64 + self.block_offsets).view(-1)
        for hot in self.caches.values():
            hot.invalidate(slots)


_CACHES: AnchorHotCaches | None = None


def build(forward_context: dict[str, Any]) -> None:
    """Build the rank's hot caches once the KV caches are bound (before any
    warmup or graph capture), replacing any earlier set."""
    global _CACHES
    _CACHES = None
    if not _ANCHORS:
        return
    layers = {
        extract_layer_index(name): (name, layer)
        for name, layer in forward_context.items()
        if hasattr(getattr(layer, "impl", None), "_reuses_topk")
    }
    first = next(iter(layers.values()))[1].kv_cache
    _CACHES = AnchorHotCaches(first.device, layers)
    assert len(_CACHES.caches) == len(_ANCHORS), "anchor hot cache layers missing"
    _extension()


def get(grace_cache: torch.Tensor) -> AnchorHotCache | None:
    """The layer's hot cache, else None (not a hot anchor)."""
    return None if _CACHES is None else _CACHES.caches.get(grace_cache.data_ptr())


def invalidate_blocks(block_ids: torch.Tensor) -> None:
    if _CACHES is not None:
        _CACHES.invalidate_blocks(block_ids)


class _Check:
    """VLLM_ANCHOR_HOT_CACHE_CHECK: every mapped call also runs FlashMLA on
    the Grace cache with the original indices; mismatching output and LSE
    elements are counted in host memory (in graphs too) and logged."""

    def __init__(self):
        import threading

        from vllm.logger import init_logger
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        self.host = torch.zeros(2, dtype=torch.int64).pin_memory()
        self.view = get_accelerator_view_from_cpu_tensor(self.host)
        self.one = torch.ones(1, dtype=torch.int64, device=self.view.device)
        log = init_logger(__name__)

        def report():
            last = None
            while True:
                threading.Event().wait(30)
                now = tuple(self.host.tolist())
                if now != last:
                    log.info("anchor hot check: calls %d, mismatches %d", *now)
                    last = now

        threading.Thread(target=report, daemon=True).start()

    def __call__(self, out, ref_out, lse, ref_lse) -> None:
        bad = (out.view(torch.int16) != ref_out.view(torch.int16)).sum()
        bad = bad + (lse.view(torch.int32) != ref_lse.view(torch.int32)).sum()
        self.view[0:1].add_(self.one)
        self.view[1:2].add_(bad)


@cache
def check() -> "_Check | None":
    return _Check() if envs.VLLM_ANCHOR_HOT_CACHE_CHECK else None


def kv_update(
    grace_cache: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
) -> None:
    if get(grace_cache) is not None:
        assert _CACHES is not None
        _CACHES.kv_update(grace_cache, kv_c_normed, k_pe, slot_mapping, k_scale)


def decodes(attn_metadata: Any) -> bool:
    assert _CACHES is not None
    return _CACHES.decodes(attn_metadata)


def stream() -> torch.cuda.Stream:
    assert _CACHES is not None
    return _CACHES.stream
```
