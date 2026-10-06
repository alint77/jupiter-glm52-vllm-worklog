# Skip-layer MLA KV on Grace, prefetched from the anchor's top-k (plan, 2026-10-06)

Status: **plan, nothing built.** Goal: move the main MLA KV of GLM-5.3's 57
index-share ("skip") layers to Grace and copy each step's selected rows into a
small HBM buffer before those layers' attention runs, so attention still reads
HBM and the step pays nothing. The freed HBM (~3 GiB per GPU) goes to hot
experts. Anchor layers (own indexer) keep their KV in HBM.

The user lifted the "KV stays on GPUs" scope rule for offload that adds no step
overhead (memory `glm-serving-scope`, 2026-10-06).

## Why this can be free when KV-on-Grace was not

- 2026-08-05-kv-grace-attn / 2026-08-06-mla-grace-kernel: FlashMLA sparse
  reading KV **directly** from Grace gets 157 GB/s (occupancy-bound, not
  link-bound): +8.5 ms/step at c4. So attention must keep reading HBM.
- The same work measured plain gathers of 656 B rows from Grace at 361-384
  GB/s and TMA at 420 GB/s: copying the rows is cheap. The copy just has to
  happen early enough.
- GLM-5.3 config `index_topk_freq=4, index_skip_topk_offset=3`
  (deepseek_v2.py:1098-1112): anchors are layers {0, 1, 2, 6, 10, ..., 74}
  (21); each anchor 4k+2 is followed by skip layers 4k+3, 4k+4, 4k+5 (19 groups,
  57 layers) that reuse its top-k. flashmla_sparse already caches the anchor's
  DCP-converted indices and reuses them in the skip layers
  (`_dcp_converted_indices`, `_reuses_topk`, flashmla_sparse.py:917-967). So a
  group's row set is known once the anchor's indexer and conversion have run,
  a full layer (~300 us) or more before the first skip layer's attention.

## Budget (per GPU, 400K, DCP4, from the 2183109 planner log)

| | today | plan |
|---|--:|--:|
| main MLA KV, 21 anchor layers (HBM) | 1.28 GiB | 1.28 GiB |
| main MLA KV, 57 skip layers | 3.49 GiB HBM | 3.49 GiB Grace |
| skip-layer HBM buffers: one shared set of 3 x R rows x 656 B (below) | 0 | 32 MB (R = 16384, worst case) |
| **freed HBM** | | **~3.4 GiB = ~170 hot experts** |

Expected gain, rough: cold experts per GPU per layer ~2.00 -> ~1.88, ~0.4
ms/step (~1.6%). To be replaced by gate 0b's replay before any build.

Copy traffic (v1, re-gather every step): 57 x U x 656 B per step, U = this
rank's union of selected rows over the 8 verify queries. U = 1000 -> 37 MB ->
~0.1 ms of link per step, issued on a side stream. v2 would copy only rows
new since the last step, but that needs a persistent buffer per layer (57,
~0.6 GiB at R = 16384) instead of one shared set; take it only if gate 0a's
churn and gate 4's trace show the v1 copies costing step time.

**One shared buffer set, single-buffered.** Main-stream order is
skip(g-1) x 3 -> anchor g -> skip(g) x 3. Group g's gather is forked from the
main stream after anchor g's index conversion, so group g-1's skip layers have
finished reading the buffers by then: one set of 3 buffers (one per position
in a group) serves all 19 groups with no double buffering. The skip layers wait
on the gather's event (e_g).

## Design

Per skip group g (anchor a, skip layers a+1..a+3), per decode step:

1. **Anchor, after `_dcp_converted_indices`** (indices: [8, 2048] global slots,
   this rank's owned slots compacted to the front, -1 tail), on a side stream:
   - **plan kernel**: unique rows U_g over the 8 queries -> buffer rows
     0..U_g-1, plus remapped indices [8, 2048] (buffer row per entry, -1 kept).
     Deterministic order: a bitmap over this rank's slot space (~100K slots,
     12.5 KB) and a prefix sum, not hashing. Slots of tokens written *this step*
     map to reserved rows (below).
   - **gather kernel**: for each of the 3 layers, copy rows U_g from that
     layer's Grace store (UVA alias) into its HBM buffer; record event e_g.
2. **Skip layer**: the KV write (`do_kv_cache_update`, fp8_ds_mla) goes to the
   Grace store at the usual slots and also into the buffer's reserved rows for
   this step's tokens (a second write with a reserved-row slot mapping; ~2 rows
   per rank). Wait e_g, then run the unchanged FlashMLA sparse decode kernel on
   (buffer, remapped indices). Same bytes and the same kernel give a
   bit-identical result.

This step's own tokens: their skip-layer KV only exists once that layer runs,
so it cannot be prefetched. They take the reserved rows (one per verify
position: 8). FlashMLA's `extra_k_cache` / `extra_indices_in_kvcache`
(flash_mla_interface.py:70-94) may express this without remapping; check
whether the sm90 fp8 sparse decode supports it in gate 1.

**Buffer size R.** Worst case U_g = 8 x 2048 = 16384 (no overlap between the 8
queries, all owned by this rank). With one shared set that is only 32 MB, so R
is the worst case and there is no overflow path.

**Other paths, correct and not optimized:**
- prefill: KV writes go to the Grace store over UVA (4K-token chunk: ~37 MB
  for 57 layers). `dcp_sparse_prefill` upconverts the rank's whole local shard,
  which now streams from Grace: 57 x L/4 x 656 B per chunk, ~2 ms at 100K and
  ~10 ms at 400K on a ~540 ms chunk. Report TTFT; prefill is not a priority.
- eager / non-graph decode: read the Grace store directly.
- prefix caching and block ids: unchanged; the Grace store is the same tensor
  shape, only allocated elsewhere.

**Where it plugs in:**
- `config/tiered_moe.py` `MLACacheTier`: new `"skip_host_uva"`.
- `v1/core/kv_cache_utils.py:1405` / `tiered_moe_kv.get_tiered_kv_memory_tier`:
  tier per KV tensor by layer (skip -> host_uva, anchor -> hbm); today it is per
  spec kind. `TieredKVCachePlan`: split main bytes into HBM / host, add buffer
  bytes to HBM, so the planner hands the rest to hot experts.
- `v1/worker/gpu/attn_utils.py:_allocate_kv_cache` (V2 runner, used in prod):
  port V1's host_uva branch (`gpu_model_runner.py:7224`: GraceAllocation,
  NUMA audit), NUMA-bound to the GPU's Grace node.
- `v1/attention/backends/mla/flashmla_sparse.py`: anchor issues plan+gather
  after conversion; skip layers swap in (buffer, remapped indices). Buffers,
  remap tensors and events are static so the decode graph captures them (the
  side-stream pattern already exists for the indexer and the in-graph cold
  prefetch).
- `mla_attention.py:611-620` (`do_kv_cache_update`): the extra reserved-row write
  for skip layers.

## Gates

**0. Measure before building (one capture run, the rest offline).**
- 0a. Index capture: env-gated dump of the anchors' DCP-converted indices per
  rank for a few hundred decode steps of agentic task-set traffic (in-graph
  copy into a preallocated device ring, D2H between steps). Report per group:
  U_g distribution (p50/p99/p99.99), step-to-step churn (rows not selected
  last step), and how often this step's own tokens are selected. This sets R,
  v1 vs v2, and the copy traffic.
- 0b. Residency replay: replay the live routing capture
  (routes-datasets/glm53-cc-20261004-job2173771) with the hot set grown by the
  freed experts per GPU; cold experts per GPU per layer -> expected ms/step at
  ~51 us per cold expert.
- **Go** if 0b gives >= 0.25 ms/step and R fits within ~0.6 GiB.

**1. Kernels, unit tests (no server).** The plan and gather kernels; test:
FlashMLA sparse decode on (buffer, remapped) equals (full cache, original
indices) bit for bit, including -1 tails, DCP compaction, reserved rows and
the worst-case U_g = R. Gather bench from a NUMA-bound Grace store: target >= 350
GB/s at 656 B rows (prior Triton gather 361-384).

**2. Allocation + planner.** The new tier on the V2 runner; the planner log
shows the skip bytes in Grace and the extra hot experts; startup and the
memory-peak check pass; prefix caching still works across turns.

**3. Exactness.** Greedy decode token-identical to prod on fixed prompts:
attention bytes and kernels are unchanged, and hot vs cold experts run the same
math. GSM8K 400 sanity.

**4. A/B on the node.** Interleaved same-node runs of prod vs skip_host_uva on
the agentic task set: step time from `VLLM_STEP_TRACE_FILE`, acceptance
(must match), a profiler trace of each (copies must not land on the critical
path: no main-stream waits on e_g), TTFT reported.

## Risks

- A copy issued inside a cold-bound MoE window takes link time from cold
  experts byte for byte: v1 ~0.1 ms/step at worst, v2 ~0. Gate 4's trace
  shows where they land.
- 57 event waits in the decode graph (one per skip layer on its group's e_g): cheap if the copies are done, each one a
  stall if not. The lead time (>= ~1 layer) vs copy time (~2 us per group at
  U = 1000) leaves wide margin.
- The FlashMLA metadata (`cache_seqlens`, tile scheduler, dummy block table)
  must accept a k_cache with R/64 blocks; check in gate 1.
- Real top-k overlap across the 8 verify queries is unknown (2026-08-05 had no
  production index trace); gate 0a measures it.
