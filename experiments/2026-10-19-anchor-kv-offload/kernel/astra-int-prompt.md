# Anchor hot cache: vLLM integration review (follow-up to your v3 kernel review)

You reviewed the v3 kernel (anchor_hot.cu); v3e adopted your P1 (cooperative
launch) and P2s. new_slots is now int64 (vLLM's slot_mapping). All kernel
checks pass (real-dump replay eager+graph, adversarial eager+graph, stress).
Booster: 10.14 / 9.70 us per call at M=16 (layer 1 / 10).

Now the integration into vLLM (diff below, against prod 24175ed0d3). Do not
run commands (your sandbox cannot run here); review the diff as given.

Context: GLM-5.3 DSA, TP4 + DCP4, MTP3 (k=3), c=4, FULL CUDA graphs for
decode (uniform decode sizes 4/8/12/16), interleaved scheduling (no mixed
prefill/decode steps). skip_host_uva (prod): the 57 index-sharing ("skip")
layers' main cache is on Grace and staged per step by SkipKVStager (keyed by
the anchor's cache data_ptr, stage() after the anchor's FlashMLA with the
anchor's converted slot indices); the 21 target anchors + MTP layer 78 keep
main cache in HBM. This change moves the 21 target anchors to Grace too
(VLLM_ANCHOR_HOT_CACHE=same|free), each with an HBM hot cache (16384 + 64
reserved rows). Decode step per anchor: KV write to Grace, then (new) the
reserved-row write of this step's tokens into the hot buffer; in
MLAAttention.forward_impl before the q absorb, start_mqa forks to a side
stream: DCP index conversion + map kernel, event record; forward_mqa waits on
the event and runs FlashMLA on the hot buffer with the mapped rows. Non-map
steps (prefill, or > max queries) read Grace directly and invalidate the
written slots in do_kv_cache_update. "same" keeps the planner's HBM charge
(so the hot expert set equals prod's; hot caches fit in the unused charge);
"free" charges only the hot caches and gives the rest to experts.

Questions:
1. Ordering/race: KV write (Grace + reserved rows, main) -> fork (side waits
   main) -> map on side -> join (main waits event) -> FlashMLA (main) ->
   stager.stage (its own stream waits main). Next step's map for the same
   layer forks after this step's FlashMLA on main. Any hazard, including
   across CUDA-graph replays and the drafter's graphs? Is record_stream on
   the side-stream-allocated conversion outputs correct/needed, including
   during graph capture?
2. Invalidation completeness: map steps rely on the kernel's new-slot
   invalidation for this step's tokens; every other step invalidates its
   whole slot_mapping in do_kv_cache_update (predicate from forward-context
   metadata, same predicate as the map). Any KV writer this misses (MTP
   drafter writes to target layers? KV transfer/offload connectors? prefix
   cache reuse of freed blocks, which reassigns physical slots without a
   write until the new owner writes, preemption/recompute, the dummy/profile
   runs, padded tokens with slot -1)? Is "slot rewritten before it is read
   again" sufficient for freshness when blocks are freed and reallocated?
3. Same ops: FlashMLA (sparse_decode_fwd with dcp_decode_sched_meta's
   schedule) on the hot buffer [257, 64, 656] with mapped rows vs on the
   Grace cache [num_blocks, 64, 656] with slots. Positions, -1 and
   multiplicity kept; valid counts and schedule computed from the slot
   indices. Anything in FlashMLA (or the non-sched _fp8_flash_mla_kernel path
   with dummy_block_table / cache_lens) that depends on the cache tensor
   size or on index values beyond row addressing?
4. Planner (tiered_moe_physical): host charge += moved anchors in both modes;
   "free" subtracts their HBM charge and adds the hot tables. Correct sizes?
   (main_cache_bytes includes the MTP layer; indexer_layer_ids includes it.)
5. The check mode (VLLM_ANCHOR_HOT_CACHE_CHECK): an extra FlashMLA on Grace
   per mapped call, bitwise compare of out (bf16 as int16) and lse (fp32 as
   int32), counts added to a pinned-host UVA tensor from inside the graph,
   logged by a thread. Is this a sound bitwise test (is FlashMLA sparse
   decode deterministic run to run with the same schedule)?
6. Anything else wrong or fragile.

## Diff
```diff
diff --git a/vllm/envs.py b/vllm/envs.py
index 20e05babcf..14101c5ec5 100644
--- a/vllm/envs.py
+++ b/vllm/envs.py
@@ -281,6 +281,10 @@ if TYPE_CHECKING:
     VLLM_DCP_SPARSE_PREFILL: bool = True
     VLLM_DCP_ONE_SHOT_COLLECTIVES: bool = False
     VLLM_DCP_SPARSE_DECODE_WIDTH: int = 0
+    VLLM_ANCHOR_HOT_CACHE: str = ""
+    VLLM_ANCHOR_HOT_CACHE_ROWS: int = 16384
+    VLLM_ANCHOR_HOT_CACHE_FORK: bool = True
+    VLLM_ANCHOR_HOT_CACHE_CHECK: bool = False
     VLLM_DCP_DECODE_SCHED: bool = True
     VLLM_SCHED_INTERLEAVE_PREFILL: int = 0
     VLLM_SCHED_INTERLEAVE_SRPT: bool = False
@@ -2016,6 +2020,23 @@ environment_variables: dict[str, Callable[[], Any]] = {
     "VLLM_DCP_SPARSE_DECODE_WIDTH": lambda: int(
         os.getenv("VLLM_DCP_SPARSE_DECODE_WIDTH", "0")
     ),
+    # Anchor layers' main MLA cache on Grace behind a per-layer HBM hot cache
+    # (skip_host_uva only): "same" keeps the planner's HBM charge for it,
+    # "free" gives that HBM to the planner; empty disables.
+    "VLLM_ANCHOR_HOT_CACHE": env_with_choices(
+        "VLLM_ANCHOR_HOT_CACHE", "", ["", "same", "free"]
+    ),
+    "VLLM_ANCHOR_HOT_CACHE_ROWS": lambda: int(
+        os.getenv("VLLM_ANCHOR_HOT_CACHE_ROWS", "16384")
+    ),
+    # map the anchor's top-k on a side stream from the start of attention
+    "VLLM_ANCHOR_HOT_CACHE_FORK": lambda: bool(
+        int(os.getenv("VLLM_ANCHOR_HOT_CACHE_FORK", "1"))
+    ),
+    # also run each mapped anchor's FlashMLA on Grace and count mismatches
+    "VLLM_ANCHOR_HOT_CACHE_CHECK": lambda: bool(
+        int(os.getenv("VLLM_ANCHOR_HOT_CACHE_CHECK", "0"))
+    ),
     # DCP fp8 sparse MLA on sm90 (non-prefill path): build FlashMLA's split
     # schedule on device from the valid counts, covering only the blocks up to
     # the longest live row and splitting them evenly, instead of FlashMLA's
@@ -2415,6 +2436,10 @@ def compile_factors() -> dict[str, object]:
         "VLLM_DCP_SPARSE_PREFILL",
         "VLLM_DCP_SPARSE_DECODE_WIDTH",
         "VLLM_DCP_DECODE_SCHED",
+        "VLLM_ANCHOR_HOT_CACHE",
+        "VLLM_ANCHOR_HOT_CACHE_ROWS",
+        "VLLM_ANCHOR_HOT_CACHE_FORK",
+        "VLLM_ANCHOR_HOT_CACHE_CHECK",
         "VLLM_SCHED_INTERLEAVE_PREFILL",
         "VLLM_SCHED_INTERLEAVE_SRPT",
         "VLLM_TIERED_MOE_KV_POOL_SEQS",
diff --git a/vllm/model_executor/layers/attention/mla_attention.py b/vllm/model_executor/layers/attention/mla_attention.py
index c394cc7a88..38e8a29e07 100644
--- a/vllm/model_executor/layers/attention/mla_attention.py
+++ b/vllm/model_executor/layers/attention/mla_attention.py
@@ -772,6 +772,9 @@ class MLAAttention(nn.Module, AttentionLayerBase):
             )
 
         if num_mqa_tokens > 0:
+            start_mqa = getattr(self.impl, "start_mqa", None)
+            if start_mqa is not None:
+                start_mqa(kv_cache, attn_metadata, num_mqa_tokens)
             mqa_q = q[:num_mqa_tokens]
             mqa_output_slice = output[:num_mqa_tokens]
 
diff --git a/vllm/model_executor/model_loader/tiered_moe_kv.py b/vllm/model_executor/model_loader/tiered_moe_kv.py
index 76f20bd202..a258dd9f11 100644
--- a/vllm/model_executor/model_loader/tiered_moe_kv.py
+++ b/vllm/model_executor/model_loader/tiered_moe_kv.py
@@ -81,6 +81,16 @@ def _glm_anchor_layers(vllm_config: "VllmConfig") -> tuple[int, ...]:
     return glm_full_indexer_layers(num_hidden_layers=78 + mtp_layers)
 
 
+def anchor_hot_cache_layers(vllm_config: "VllmConfig") -> tuple[int, ...]:
+    """Anchors whose main cache moves to Grace under VLLM_ANCHOR_HOT_CACHE:
+    the target's, not a grafted MTP layer's."""
+    if not envs.VLLM_ANCHOR_HOT_CACHE:
+        return ()
+    if vllm_config.tiered_moe_config.mla_cache_tier != "skip_host_uva":
+        raise ValueError("VLLM_ANCHOR_HOT_CACHE requires skip_host_uva")
+    return glm_full_indexer_layers(num_hidden_layers=78)
+
+
 def get_tiered_kv_memory_tier(
     vllm_config: "VllmConfig", spec: KVCacheSpec, layer_name: str | None = None
 ) -> tuple[str, str]:
@@ -96,12 +106,12 @@ def get_tiered_kv_memory_tier(
     if tiered.mla_cache_tier == "skip_host_uva":
         from vllm.model_executor.models.utils import extract_layer_index
 
-        if (
-            spec_kind == "main"
-            and layer_name is not None
-            and extract_layer_index(layer_name) not in _glm_anchor_layers(vllm_config)
-        ):
-            return "host_uva", "grace_numa_node"
+        if spec_kind == "main" and layer_name is not None:
+            index = extract_layer_index(layer_name)
+            if index not in _glm_anchor_layers(
+                vllm_config
+            ) or index in anchor_hot_cache_layers(vllm_config):
+                return "host_uva", "grace_numa_node"
         return "hbm", "gpu_rank"
     if tiered.mla_cache_tier != "host_uva":
         raise ValueError("Tiered KV allocation requires an explicit cache tier")
diff --git a/vllm/model_executor/model_loader/tiered_moe_physical.py b/vllm/model_executor/model_loader/tiered_moe_physical.py
index 26dba54c59..c72d36739e 100644
--- a/vllm/model_executor/model_loader/tiered_moe_physical.py
+++ b/vllm/model_executor/model_loader/tiered_moe_physical.py
@@ -319,10 +319,28 @@ def plan_tiered_moe_scenario(
     fixed_hbm_allocations["tiered_moe_runtime_buffers"] = (
         runtime_buffers.steady_hbm_bytes
     )
-    if kv_plan.main_hbm_bytes:
-        fixed_hbm_allocations["main_mla_cache"] = kv_plan.main_hbm_bytes
-    if kv_plan.host_bytes:
-        fixed_host_allocations["host_mla_cache"] = kv_plan.host_bytes
+    main_hbm_bytes, host_bytes = kv_plan.main_hbm_bytes, kv_plan.host_bytes
+    if envs.VLLM_ANCHOR_HOT_CACHE and cache_tier == "skip_host_uva":
+        from vllm.v1.attention.backends.mla import anchor_hot_cache, skip_kv_stage
+
+        # the target's anchors move to Grace; "same" keeps their HBM charge
+        # (the hot set is skip_host_uva's), "free" charges their hot caches
+        moved = len(kv_plan.indexer_layer_ids) - num_mtp_layers
+        per_layer = kv_plan.main_cache_bytes // kv_plan.main_layer_count
+        host_bytes += moved * per_layer
+        if envs.VLLM_ANCHOR_HOT_CACHE == "free":
+            queries, columns = skip_kv_stage.limits()
+            main_hbm_bytes -= moved * per_layer
+            fixed_hbm_allocations["anchor_hot_cache"] = moved * (
+                anchor_hot_cache.table_bytes(
+                    per_layer // kv_plan.main_page_bytes_per_layer * kv_plan.block_size,
+                    queries * columns,
+                )
+            )
+    if main_hbm_bytes:
+        fixed_hbm_allocations["main_mla_cache"] = main_hbm_bytes
+    if host_bytes:
+        fixed_host_allocations["host_mla_cache"] = host_bytes
     if cache_tier == "skip_host_uva":
         from vllm.v1.attention.backends.mla.skip_kv_stage import staging_bytes
 
diff --git a/vllm/v1/attention/backends/mla/anchor_hot_cache.py b/vllm/v1/attention/backends/mla/anchor_hot_cache.py
new file mode 100644
index 0000000000..c2584e59d2
--- /dev/null
+++ b/vllm/v1/attention/backends/mla/anchor_hot_cache.py
@@ -0,0 +1,303 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""HBM hot cache for anchor layers whose main MLA cache lives on Grace.
+
+With ``VLLM_ANCHOR_HOT_CACHE`` (``same`` or ``free``) under ``skip_host_uva``
+the target's anchor layers (those with their own DSA indexer) keep their
+fp8_ds_mla main cache in pinned Grace memory too; the grafted MTP layer stays
+in HBM. Each such layer keeps ``VLLM_ANCHOR_HOT_CACHE_ROWS`` rows of it in an
+HBM buffer, plus RESERVED_ROWS rows for this step's tokens.
+
+In a decode step the anchor's converted top-k (this rank's slots) goes
+through one cooperative kernel (anchor_hot.cu): a hit keeps its row, a miss
+takes a CLOCK victim and is copied from Grace, and this step's token j maps to
+row ``rows + j``, which the layer's KV write fills. FlashMLA then reads the
+buffer with the mapped indices. Entries keep position, multiplicity and -1, so
+it reads the same bytes in the same order. Any other step reads Grace
+directly and drops the slots it writes from the buffer.
+
+``same`` keeps the planner's HBM charge for these caches, so the hot expert
+set is skip_host_uva's; ``free`` hands the freed HBM to the planner.
+"""
+
+from functools import cache
+from pathlib import Path
+from typing import Any
+
+import torch
+
+import vllm.envs as envs
+from vllm.forward_context import get_forward_context
+from vllm.model_executor.models.utils import extract_layer_index
+from vllm.triton_utils import tl, triton
+
+_HERE = Path(__file__).resolve().parent
+RESERVED_ROWS = 64  # one per query token of a step (anchor_hot.cu: t <= 64)
+ROW_BYTES = 656
+_CTL = 32
+_ANCHORS: frozenset[int] = frozenset()
+
+
+def enabled() -> bool:
+    return envs.VLLM_ANCHOR_HOT_CACHE in ("same", "free")
+
+
+def configure(vllm_config) -> None:
+    """Record the hot anchors while the config is set (layer init)."""
+    global _ANCHORS
+    from vllm.model_executor.model_loader.tiered_moe_kv import (
+        anchor_hot_cache_layers,
+    )
+
+    _ANCHORS = frozenset(anchor_hot_cache_layers(vllm_config))
+
+
+def table_bytes(num_slots: int, max_entries: int) -> int:
+    """HBM one layer's hot cache holds: buffer, slot/row maps, miss lists."""
+    rows = envs.VLLM_ANCHOR_HOT_CACHE_ROWS
+    return (
+        (rows + RESERVED_ROWS) * ROW_BYTES
+        + 4 * (num_slots + 2 * rows + _CTL + 3 * max_entries)
+        + 8 * RESERVED_ROWS
+    )
+
+
+@cache
+def _extension():
+    from torch.utils.cpp_extension import load
+
+    name = "vllm_anchor_hot"
+    build = Path(envs.VLLM_CACHE_ROOT) / "torch_extensions" / name
+    build.mkdir(parents=True, exist_ok=True)
+    return load(
+        name=name,
+        sources=[str(_HERE / "anchor_hot.cu")],
+        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
+        build_directory=str(build),
+        verbose=False,
+    )
+
+
+@triton.jit
+def _invalidate_kernel(slots_ptr, n, where_ptr, slot_of_ptr, BLOCK: tl.constexpr):
+    """Drop every written slot from the buffer (both maps)."""
+    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
+    s = tl.load(slots_ptr + offs, mask=offs < n, other=-1)
+    live = s >= 0
+    s = tl.where(live, s, 0)
+    r = tl.load(where_ptr + s, mask=live, other=-1)
+    tl.store(slot_of_ptr + tl.where(r >= 0, r, 0), -1, mask=live & (r >= 0))
+    tl.store(where_ptr + s, -1, mask=live)
+
+
+class AnchorHotCache:
+    """One anchor layer's buffer and maps (anchor_hot.cu's State)."""
+
+    def __init__(self, grace_cache: torch.Tensor, max_entries: int):
+        rows = envs.VLLM_ANCHOR_HOT_CACHE_ROWS
+        # the kernel's sweep window and victim guarantee (anchor_hot_map)
+        assert rows % 64 == 0 and rows >= 4096 and max_entries <= rows
+        device = grace_cache.device
+        self.grace = grace_cache
+        self.rows = rows
+        self.max_entries = max_entries
+        num_slots = grace_cache.numel() // ROW_BYTES
+        self.buffer = torch.zeros(
+            (rows + RESERVED_ROWS) // 64,
+            64,
+            ROW_BYTES,
+            dtype=torch.uint8,
+            device=device,
+        )
+        i32 = dict(dtype=torch.int32, device=device)
+        self.where = torch.full((num_slots,), -1, **i32)
+        self.slot_of = torch.full((rows,), -1, **i32)
+        self.status = torch.zeros(rows, **i32)
+        self.ctl = torch.zeros(_CTL, **i32)
+        self.ctl[10:12] = -1
+        self.miss = torch.zeros(max_entries, **i32)
+        self.miss_row = torch.zeros(max_entries, **i32)
+        self.out = torch.full((max_entries,), -1, **i32)
+        self.ready = torch.cuda.Event()
+
+    def map(self, indices: torch.Tensor, new_slots: torch.Tensor) -> torch.Tensor:
+        """Buffer rows for ``indices`` [T, W] (this rank's slots or -1);
+        ``new_slots``: this step's slot mapping [T]."""
+        n = indices.numel()
+        out = self.out[:n]
+        _extension().anchor_hot_map(
+            indices.reshape(-1),
+            new_slots,
+            out,
+            self.grace,
+            self.buffer,
+            self.where,
+            self.slot_of,
+            self.status,
+            self.ctl,
+            self.miss,
+            self.miss_row,
+        )
+        return out.view(indices.shape)
+
+    def invalidate(self, slot_mapping: torch.Tensor) -> None:
+        n = slot_mapping.numel()
+        if n:
+            _invalidate_kernel[(triton.cdiv(n, 1024),)](
+                slot_mapping, n, self.where, self.slot_of, BLOCK=1024
+            )
+
+
+class AnchorHotCaches:
+    """The rank's hot caches, keyed by the anchors' Grace cache pointers."""
+
+    def __init__(self, device: torch.device, layers: dict[int, Any]):
+        from vllm.v1.attention.backends.mla import skip_kv_stage
+
+        queries, columns = skip_kv_stage.limits()
+        self.max_queries = queries
+        self.caches: dict[int, AnchorHotCache] = {}
+        self.names: dict[int, str] = {}
+        for index in sorted(layers):
+            name, layer = layers[index]
+            if index in _ANCHORS and not layer.impl._reuses_topk:
+                ptr = layer.kv_cache.data_ptr()
+                self.caches[ptr] = AnchorHotCache(layer.kv_cache, queries * columns)
+                self.names[ptr] = name
+        self.stream = torch.cuda.Stream(device)
+        self.first = next(iter(self.caches), None)
+        rows = envs.VLLM_ANCHOR_HOT_CACHE_ROWS
+        self._reserved = rows + torch.arange(
+            RESERVED_ROWS, dtype=torch.int64, device=device
+        )
+        self.reserved_map = torch.full_like(self._reserved, -1)
+
+    def decodes(self, attn_metadata: Any) -> bool:
+        """Whether the step maps through the hot cache (else Grace is read
+        directly and written slots are dropped)."""
+        return (
+            attn_metadata is not None
+            and attn_metadata.num_prefills == 0
+            and attn_metadata.num_actual_tokens <= self.max_queries
+        )
+
+    def kv_update(
+        self,
+        grace_cache: torch.Tensor,
+        kv_c_normed: torch.Tensor,
+        k_pe: torch.Tensor,
+        slot_mapping: torch.Tensor,
+        k_scale: torch.Tensor,
+    ) -> None:
+        """Anchor's KV write (after Grace's): in a mapped step this step's
+        tokens also go to its reserved rows (rows + j for token j, skipped
+        where another DCP rank owns it); in any other step the written slots
+        leave the buffer."""
+        ptr = grace_cache.data_ptr()
+        hot = self.caches.get(ptr)
+        if hot is None:
+            return
+        md = get_forward_context().attn_metadata
+        if isinstance(md, list):
+            md = md[0]
+        if isinstance(md, dict):
+            md = md.get(self.names[ptr])
+        if not self.decodes(md):
+            hot.invalidate(slot_mapping)
+            return
+        t = slot_mapping.numel()
+        assert t <= RESERVED_ROWS
+        from vllm import _custom_ops as ops
+
+        if ptr == self.first:  # the same for every anchor
+            self.reserved_map[:t] = torch.where(
+                slot_mapping >= 0, self._reserved[:t], -1
+            )
+        ops.concat_and_cache_mla(
+            kv_c_normed,
+            k_pe.squeeze(1),
+            hot.buffer,
+            self.reserved_map[:t],
+            kv_cache_dtype="fp8_ds_mla",
+            scale=k_scale,
+        )
+
+
+_CACHES: AnchorHotCaches | None = None
+
+
+def get(grace_cache: torch.Tensor) -> AnchorHotCache | None:
+    """The layer's hot cache, else None (not a hot anchor, or caches unbound).
+    The rank's caches are built from the bound attention layers on first use."""
+    global _CACHES
+    if _CACHES is None:
+        if grace_cache.numel() == 0:
+            return None
+        layers = {
+            extract_layer_index(name): (name, layer)
+            for name, layer in get_forward_context().no_compile_layers.items()
+            if hasattr(getattr(layer, "impl", None), "_reuses_topk")
+        }
+        _CACHES = AnchorHotCaches(grace_cache.device, layers)
+    return _CACHES.caches.get(grace_cache.data_ptr())
+
+
+class _Check:
+    """VLLM_ANCHOR_HOT_CACHE_CHECK: every mapped call also runs FlashMLA on
+    the Grace cache with the original indices; mismatching output and LSE
+    elements are counted in host memory (in graphs too) and logged."""
+
+    def __init__(self):
+        import threading
+
+        from vllm.logger import init_logger
+        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
+
+        self.host = torch.zeros(2, dtype=torch.int64).pin_memory()
+        self.view = get_accelerator_view_from_cpu_tensor(self.host)
+        self.one = torch.ones(1, dtype=torch.int64, device=self.view.device)
+        log = init_logger(__name__)
+
+        def report():
+            last = None
+            while True:
+                threading.Event().wait(30)
+                now = tuple(self.host.tolist())
+                if now != last:
+                    log.info("anchor hot check: calls %d, mismatches %d", *now)
+                    last = now
+
+        threading.Thread(target=report, daemon=True).start()
+
+    def __call__(self, out, ref_out, lse, ref_lse) -> None:
+        bad = (out.view(torch.int16) != ref_out.view(torch.int16)).sum()
+        bad = bad + (lse.view(torch.int32) != ref_lse.view(torch.int32)).sum()
+        self.view[0:1].add_(self.one)
+        self.view[1:2].add_(bad)
+
+
+@cache
+def check() -> "_Check | None":
+    return _Check() if envs.VLLM_ANCHOR_HOT_CACHE_CHECK else None
+
+
+def kv_update(
+    grace_cache: torch.Tensor,
+    kv_c_normed: torch.Tensor,
+    k_pe: torch.Tensor,
+    slot_mapping: torch.Tensor,
+    k_scale: torch.Tensor,
+) -> None:
+    if get(grace_cache) is not None:
+        assert _CACHES is not None
+        _CACHES.kv_update(grace_cache, kv_c_normed, k_pe, slot_mapping, k_scale)
+
+
+def decodes(attn_metadata: Any) -> bool:
+    assert _CACHES is not None
+    return _CACHES.decodes(attn_metadata)
+
+
+def stream() -> torch.cuda.Stream:
+    assert _CACHES is not None
+    return _CACHES.stream
diff --git a/vllm/v1/attention/backends/mla/flashmla_sparse.py b/vllm/v1/attention/backends/mla/flashmla_sparse.py
index 19fa807304..84b09a99c4 100644
--- a/vllm/v1/attention/backends/mla/flashmla_sparse.py
+++ b/vllm/v1/attention/backends/mla/flashmla_sparse.py
@@ -28,7 +28,7 @@ from vllm.v1.attention.backend import (
     CommonAttentionMetadata,
     MultipleOf,
 )
-from vllm.v1.attention.backends.mla import skip_kv_stage
+from vllm.v1.attention.backends.mla import anchor_hot_cache, skip_kv_stage
 from vllm.v1.attention.backends.mla.sparse_utils import (
     dcp_decode_sched_meta,
     triton_convert_req_index_to_global_index,
@@ -694,6 +694,16 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
         vllm_config = get_current_vllm_config()
         # index-share layers' main cache on Grace, staged into HBM per decode
         self._skip_kv = vllm_config.tiered_moe_config.mla_cache_tier == "skip_host_uva"
+        # anchor's main cache on Grace behind an HBM hot cache (the grafted MTP
+        # layer has none: anchor_hot_cache.get returns None for it)
+        self._hot_anchor = (
+            self._skip_kv and not self._reuses_topk and anchor_hot_cache.enabled()
+        )
+        if self._hot_anchor:
+            anchor_hot_cache.configure(vllm_config)
+        # (indices, valid counts, schedule, hot rows, ready event) mapped on the
+        # side stream by start_mqa for this layer's forward_mqa
+        self._hot_pending: tuple | None = None
         # Prefill BF16 kernel requires 64 on Hopper, 128 on Blackwell
         self.prefill_padding = (
             128 if current_platform.is_device_capability_family(100) else 64
@@ -892,20 +902,28 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
             return self._forward_dcp_sparse_prefill(
                 q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
             )
-        valid_counts: torch.Tensor | None = None
-        sched: tuple[torch.Tensor, torch.Tensor] | None = None
-        if self.dcp_world_size > 1:
-            topk_indices, valid_counts, sched = self._dcp_converted_indices(
-                topk_indices, attn_metadata, decode_sched=self._dcp_decode_sched(q)
+        grace_cache = kv_c_and_k_pe_cache
+        hot_rows: torch.Tensor | None = None
+        pending, self._hot_pending = self._hot_pending, None
+        if pending is not None:
+            topk_indices, valid_counts, sched, hot_rows, ready = pending
+            assert (sched is not None) == (
+                self.dcp_world_size > 1 and self._dcp_decode_sched(q)
             )
+            torch.cuda.current_stream().wait_event(ready)
         else:
-            topk_indices = triton_convert_req_index_to_global_index(
-                attn_metadata.req_id_per_token,
-                attn_metadata.block_table,
-                topk_indices,
-                BLOCK_SIZE=attn_metadata.block_size,
-                NUM_TOPK_TOKENS=topk_indices.shape[1],
+            topk_indices, valid_counts, sched = self._converted_indices(
+                topk_indices, attn_metadata, self._dcp_decode_sched(q)
             )
+            hot = self._hot_cache(kv_c_and_k_pe_cache, attn_metadata)
+            if hot is not None:
+                hot_rows = hot.map(
+                    topk_indices, attn_metadata.slot_mapping[: q.shape[0]]
+                )
+        if hot_rows is not None:
+            hot = anchor_hot_cache.get(grace_cache)
+            assert hot is not None
+            kv_c_and_k_pe_cache = hot.buffer
         stager = (
             skip_kv_stage.get_stager(q.device)
             if self._skip_kv
@@ -926,11 +944,12 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
         )
         fp8_metadata = attn_metadata.fp8_extra_metadata
 
+        attn_indices = topk_indices if hot_rows is None else hot_rows
         if sched is not None:
             _attn_out, _lse, _, _ = torch.ops._flashmla_C.sparse_decode_fwd(
                 q.unsqueeze(0),
                 kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
-                topk_indices.unsqueeze(0),
+                attn_indices.unsqueeze(0),
                 None,
                 None,
                 *sched,
@@ -945,13 +964,31 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
             _attn_out, _lse = self._fp8_flash_mla_kernel(
                 q=q.unsqueeze(0),  # add batch_dim: (T, H, D) -> (1, T, H, D)
                 kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
-                topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
+                topk_indices=attn_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
                 kernel_metadata=fp8_metadata,
             )
 
+        if hot_rows is not None and (check := anchor_hot_cache.check()) is not None:
+            assert sched is not None
+            ref_out, ref_lse, _, _ = torch.ops._flashmla_C.sparse_decode_fwd(
+                q.unsqueeze(0),
+                grace_cache.view(torch.uint8).unsqueeze(-2),
+                topk_indices.unsqueeze(0),
+                None,
+                None,
+                *sched,
+                None,
+                None,
+                None,
+                512,
+                self.softmax_scale,
+                None,
+            )
+            check(_attn_out, ref_out, _lse, ref_lse)
+
         if stager is not None and not self._reuses_topk:
             # after the anchor's own attention, so the copy does not slow it
-            stager.stage(kv_c_and_k_pe_cache, topk_indices, attn_metadata.slot_mapping)
+            stager.stage(grace_cache, topk_indices, attn_metadata.slot_mapping)
 
         # Output is (1, T, H, D_v) -> (T, H, D_v); lse is (1, H, T) -> (T, H)
         lse = _lse.squeeze(0).transpose(0, 1)
@@ -1000,6 +1037,10 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
         super().do_kv_cache_update(
             kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
         )
+        if self._hot_anchor:
+            anchor_hot_cache.kv_update(
+                kv_cache, kv_c_normed, k_pe, slot_mapping, k_scale
+            )
         if (
             self._skip_kv
             and self._reuses_topk
@@ -1009,16 +1050,80 @@ class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
                 kv_cache, kv_c_normed, k_pe, slot_mapping, k_scale
             )
 
-    def _dcp_decode_sched(self, q: torch.Tensor) -> bool:
-        """Whether this decode call can run on dcp_decode_sched_meta's schedule
-        (FlashMLA sm90 fp8 sparse decode, no head padding)."""
+    def _dcp_decode_sched(self, q: torch.Tensor | int) -> bool:
+        """Whether this decode call (q, or its head count) can run on
+        dcp_decode_sched_meta's schedule (FlashMLA sm90 fp8 sparse decode, no
+        head padding)."""
+        heads = q if isinstance(q, int) else q.shape[1]
         return (
             envs.VLLM_DCP_DECODE_SCHED
             and current_platform.is_device_capability(90)
-            and q.shape[1] == self.fp8_decode_padded_heads
+            and heads == self.fp8_decode_padded_heads
             and hasattr(torch.ops._flashmla_C, "sparse_decode_fwd")
         )
 
+    def _converted_indices(
+        self,
+        topk_indices: torch.Tensor,
+        attn_metadata: FlashMLASparseMetadata,
+        decode_sched: bool,
+    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple | None]:
+        """Per-request top-k to this rank's slots (decode or mixed batch)."""
+        if self.dcp_world_size > 1:
+            return self._dcp_converted_indices(
+                topk_indices, attn_metadata, decode_sched=decode_sched
+            )
+        indices = triton_convert_req_index_to_global_index(
+            attn_metadata.req_id_per_token,
+            attn_metadata.block_table,
+            topk_indices,
+            BLOCK_SIZE=attn_metadata.block_size,
+            NUM_TOPK_TOKENS=topk_indices.shape[1],
+        )
+        return indices, None, None
+
+    def _hot_cache(
+        self, kv_cache: torch.Tensor, attn_metadata: FlashMLASparseMetadata
+    ) -> "anchor_hot_cache.AnchorHotCache | None":
+        if not self._hot_anchor:
+            return None
+        hot = anchor_hot_cache.get(kv_cache)
+        if hot is None or not anchor_hot_cache.decodes(attn_metadata):
+            return None
+        return hot
+
+    def start_mqa(
+        self,
+        kv_cache: torch.Tensor,
+        attn_metadata: FlashMLASparseMetadata,
+        num_tokens: int,
+    ) -> None:
+        """Start of the decode attention, before the query absorb and its
+        all-gather: convert and map this anchor's top-k on the hot cache's
+        stream, so the copy from Grace overlaps them."""
+        if not envs.VLLM_ANCHOR_HOT_CACHE_FORK:
+            return
+        hot = self._hot_cache(kv_cache, attn_metadata)
+        if hot is None or not attn_metadata.fp8_use_mixed_batch:
+            return
+        assert self.topk_indices_buffer is not None
+        main = torch.cuda.current_stream()
+        side = anchor_hot_cache.stream()
+        side.wait_stream(main)
+        with torch.cuda.stream(side):
+            indices, valid_counts, sched = self._converted_indices(
+                self.topk_indices_buffer[:num_tokens],
+                attn_metadata,
+                self._dcp_decode_sched(self.num_heads * self.dcp_world_size),
+            )
+            rows = hot.map(indices, attn_metadata.slot_mapping[:num_tokens])
+            hot.ready.record(side)
+        # allocated on the side stream, read on main
+        for t in (indices, valid_counts, *(sched or ())):
+            if t is not None:
+                t.record_stream(main)
+        self._hot_pending = (indices, valid_counts, sched, rows, hot.ready)
+
     def _dcp_converted_indices(
         self,
         topk_indices: torch.Tensor,
```

## Unchanged context: vllm/v1/attention/backends/mla/skip_kv_stage.py (SkipKVStager)
```python
class SkipKVStager:
    """Per-rank staging state shared by the attention layers on the rank."""

    def __init__(self, device: torch.device, layers: dict[int, Any]):
        self.max_queries, self.columns = limits()
        self.gather_rows = gather_rows()
        rows = self.gather_rows + RESERVED_ROWS
        self.buffers = torch.zeros(
            STAGE_BUFFERS, rows // 64, 64, ROW_BYTES, dtype=torch.uint8, device=device
        )
        self.slots = torch.zeros(self.gather_rows, dtype=torch.int32, device=device)
        self.count = torch.zeros(1, dtype=torch.int32, device=device)
        self.rows = torch.full(
            (self.max_queries, self.columns), -1, dtype=torch.int32, device=device
        )
        self.stream = torch.cuda.Stream(device)
        # anchor cache -> (its skip layers' caches, ready event, group number);
        # skip cache -> buffer position
        self.groups: dict[int, tuple[list[torch.Tensor], torch.cuda.Event, int]]
        self.groups = {}
        self.position: dict[int, int] = {}
        self._ready: torch.cuda.Event | None = None
        self._width = self.columns  # the anchor's index width, set by stage()
        self._reserved = self.gather_rows + torch.arange(
            self.max_queries, dtype=torch.int64, device=device
        )
        self.reserved_map = torch.full_like(self._reserved, -1)
        num_slots = 0
        for index, layer in layers.items():
            if layer.impl._reuses_topk:
                continue
            group: list[torch.Tensor] = []
            while index + 1 + len(group) in layers and (
                layers[index + 1 + len(group)].impl._reuses_topk
            ):
                skip = layers[index + 1 + len(group)].kv_cache
                self.position[skip.data_ptr()] = len(group)
                group.append(skip)
                num_slots = skip.numel() // ROW_BYTES
            assert len(group) <= STAGE_BUFFERS
            if group:
                self.groups[layer.kv_cache.data_ptr()] = (
                    group,
                    torch.cuda.Event(),
                    len(self.groups),
                )
        # slot -> staged buffer row, -2 while marked, -1 otherwise (the last
        # entry is unused padding)
        self.row_of = torch.full((num_slots + 1,), -1, dtype=torch.int32, device=device)
        self.row_of[num_slots] = -3
        # VLLM_SKIP_KV_STATS: per group, a histogram of staged rows in 256-row bins
        self.hist = (
            torch.zeros(len(self.groups), self.gather_rows // 256 + 1, device=device)
            if os.environ.get("VLLM_SKIP_KV_STATS")
            else None
        )

    def stage(
        self, anchor_cache: torch.Tensor, indices: torch.Tensor, new_slots: torch.Tensor
    ) -> None:
        """Anchor, after its FlashMLA launch: start its group's copy."""
        entry = self.groups.get(anchor_cache.data_ptr())
        if entry is None:
            return
        group, ready, number = entry
        t, n = indices.shape[0], indices.numel()
        # every entry may be a distinct row: the buffers must hold them all
        assert t <= self.max_queries and indices.shape[1] <= self.columns, (
            f"skip-KV staging sized for {self.max_queries} x {self.columns}, "
            f"got {tuple(indices.shape)}"
        )
        self._width = indices.shape[1]
        new = new_slots[:t]
        if number == 0:  # this step's tokens are the same for every group
            self.reserved_map[:t] = torch.where(new >= 0, self._reserved[:t], -1)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.count.zero_()
            grid = (triton.cdiv(n, 1024),)
            _mark_rows_kernel[grid](indices, n, new, self.row_of, T=t, BLOCK=1024)
            num_slots = self.row_of.numel() - 1
            _compact_rows_kernel[(triton.cdiv(num_slots, 4096),)](
                self.row_of, num_slots, self.slots, self.count, CHUNK=4096
            )
            _remap_rows_kernel[grid](
                indices,
                n,
                new,
                self.row_of,
                self.rows,
                T=t,
                RESERVED=self.gather_rows,
                BLOCK=1024,
            )
            srcs = [c.view(torch.int32) for c in group]
            srcs += [srcs[0]] * (STAGE_BUFFERS - len(srcs))
            words = ROW_BYTES // 4
            _gather_rows_kernel[(GATHER_CTAS, len(group))](
                *srcs,
                self.buffers.view(torch.int32),
                self.slots,
                self.count,
                self.row_of,
                DST_WORDS=(self.gather_rows + RESERVED_ROWS) * words,
                ROW_WORDS=words,
                BLOCK_R=32,
                BLOCK_W=triton.next_power_of_2(words),
            )
            if self.hist is not None:
                self.hist[number].index_add_(
                    0, (self.count // 256).long(), self.count.new_ones(1).float()
                )
            ready.record(self.stream)
        self._ready = ready

    def staged(
        self, cache: torch.Tensor, num_queries: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Skip layer: its staged buffer and remapped indices, else None."""
        p = self.position.get(cache.data_ptr())
        if p is None:
            return None
        assert self._ready is not None
        torch.cuda.current_stream().wait_event(self._ready)
        rows = self.rows.view(-1)[: num_queries * self._width]
        return self.buffers[p], rows.view(num_queries, self._width)

    def write_new(
        self,
        cache: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> None:
        """Skip layer's KV write: this step's tokens into its reserved rows
        (the mapping is built by the anchor's ``stage``)."""
        p = self.position.get(cache.data_ptr())
        if p is None:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            self.buffers[p],
            self.reserved_map[: slot_mapping.numel()],
            kv_cache_dtype="fp8_ds_mla",
            scale=k_scale,
        )


_STAGER: SkipKVStager | None = None
```

## New file in full: anchor_hot_cache.py
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HBM hot cache for anchor layers whose main MLA cache lives on Grace.

With ``VLLM_ANCHOR_HOT_CACHE`` (``same`` or ``free``) under ``skip_host_uva``
the target's anchor layers (those with their own DSA indexer) keep their
fp8_ds_mla main cache in pinned Grace memory too; the grafted MTP layer stays
in HBM. Each such layer keeps ``VLLM_ANCHOR_HOT_CACHE_ROWS`` rows of it in an
HBM buffer, plus RESERVED_ROWS rows for this step's tokens.

In a decode step the anchor's converted top-k (this rank's slots) goes
through one cooperative kernel (anchor_hot.cu): a hit keeps its row, a miss
takes a CLOCK victim and is copied from Grace, and this step's token j maps to
row ``rows + j``, which the layer's KV write fills. FlashMLA then reads the
buffer with the mapped indices. Entries keep position, multiplicity and -1, so
it reads the same bytes in the same order. Any other step reads Grace
directly and drops the slots it writes from the buffer.

``same`` keeps the planner's HBM charge for these caches, so the hot expert
set is skip_host_uva's; ``free`` hands the freed HBM to the planner.
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
_ANCHORS: frozenset[int] = frozenset()


def enabled() -> bool:
    return envs.VLLM_ANCHOR_HOT_CACHE in ("same", "free")


def configure(vllm_config) -> None:
    """Record the hot anchors while the config is set (layer init)."""
    global _ANCHORS
    from vllm.model_executor.model_loader.tiered_moe_kv import (
        anchor_hot_cache_layers,
    )

    _ANCHORS = frozenset(anchor_hot_cache_layers(vllm_config))


def table_bytes(num_slots: int, max_entries: int) -> int:
    """HBM one layer's hot cache holds: buffer, slot/row maps, miss lists."""
    rows = envs.VLLM_ANCHOR_HOT_CACHE_ROWS
    return (
        (rows + RESERVED_ROWS) * ROW_BYTES
        + 4 * (num_slots + 2 * rows + _CTL + 3 * max_entries)
        + 8 * RESERVED_ROWS
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

    def __init__(self, grace_cache: torch.Tensor, max_entries: int):
        rows = envs.VLLM_ANCHOR_HOT_CACHE_ROWS
        # the kernel's sweep window and victim guarantee (anchor_hot_map)
        assert rows % 64 == 0 and rows >= 4096 and max_entries <= rows
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
        self.ready = torch.cuda.Event()

    def map(self, indices: torch.Tensor, new_slots: torch.Tensor) -> torch.Tensor:
        """Buffer rows for ``indices`` [T, W] (this rank's slots or -1);
        ``new_slots``: this step's slot mapping [T]."""
        n = indices.numel()
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

    def invalidate(self, slot_mapping: torch.Tensor) -> None:
        n = slot_mapping.numel()
        if n:
            _invalidate_kernel[(triton.cdiv(n, 1024),)](
                slot_mapping, n, self.where, self.slot_of, BLOCK=1024
            )


class AnchorHotCaches:
    """The rank's hot caches, keyed by the anchors' Grace cache pointers."""

    def __init__(self, device: torch.device, layers: dict[int, Any]):
        from vllm.v1.attention.backends.mla import skip_kv_stage

        queries, columns = skip_kv_stage.limits()
        self.max_queries = queries
        self.caches: dict[int, AnchorHotCache] = {}
        self.names: dict[int, str] = {}
        for index in sorted(layers):
            name, layer = layers[index]
            if index in _ANCHORS and not layer.impl._reuses_topk:
                ptr = layer.kv_cache.data_ptr()
                self.caches[ptr] = AnchorHotCache(layer.kv_cache, queries * columns)
                self.names[ptr] = name
        self.stream = torch.cuda.Stream(device)
        self.first = next(iter(self.caches), None)
        rows = envs.VLLM_ANCHOR_HOT_CACHE_ROWS
        self._reserved = rows + torch.arange(
            RESERVED_ROWS, dtype=torch.int64, device=device
        )
        self.reserved_map = torch.full_like(self._reserved, -1)

    def decodes(self, attn_metadata: Any) -> bool:
        """Whether the step maps through the hot cache (else Grace is read
        directly and written slots are dropped)."""
        return (
            attn_metadata is not None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_actual_tokens <= self.max_queries
        )

    def kv_update(
        self,
        grace_cache: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> None:
        """Anchor's KV write (after Grace's): in a mapped step this step's
        tokens also go to its reserved rows (rows + j for token j, skipped
        where another DCP rank owns it); in any other step the written slots
        leave the buffer."""
        ptr = grace_cache.data_ptr()
        hot = self.caches.get(ptr)
        if hot is None:
            return
        md = get_forward_context().attn_metadata
        if isinstance(md, list):
            md = md[0]
        if isinstance(md, dict):
            md = md.get(self.names[ptr])
        if not self.decodes(md):
            hot.invalidate(slot_mapping)
            return
        t = slot_mapping.numel()
        assert t <= RESERVED_ROWS
        from vllm import _custom_ops as ops

        if ptr == self.first:  # the same for every anchor
            self.reserved_map[:t] = torch.where(
                slot_mapping >= 0, self._reserved[:t], -1
            )
        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            hot.buffer,
            self.reserved_map[:t],
            kv_cache_dtype="fp8_ds_mla",
            scale=k_scale,
        )


_CACHES: AnchorHotCaches | None = None


def get(grace_cache: torch.Tensor) -> AnchorHotCache | None:
    """The layer's hot cache, else None (not a hot anchor, or caches unbound).
    The rank's caches are built from the bound attention layers on first use."""
    global _CACHES
    if _CACHES is None:
        if grace_cache.numel() == 0:
            return None
        layers = {
            extract_layer_index(name): (name, layer)
            for name, layer in get_forward_context().no_compile_layers.items()
            if hasattr(getattr(layer, "impl", None), "_reuses_topk")
        }
        _CACHES = AnchorHotCaches(grace_cache.device, layers)
    return _CACHES.caches.get(grace_cache.data_ptr())


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
