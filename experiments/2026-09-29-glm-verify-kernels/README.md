# GLM-5.3 verify-graph kernel work (DFlash2 k=7, DCP4, 400K, c=1)

Goal: cut decode step time with execution-only changes (same ops, same math,
no quantization, DCP4 kept). Baseline: serve.sh defaults (DFlash2 k=7 eager
drafter, DCP4, reserve 7, prefix caching), task-set step ~26.0 ms.

## Real host overhead: nsys, graph-level trace (job 2111454)

The torch-profiler traces (2108844) showed ~18% of steps stalling 1-11 ms at
the verify graph's first all-reduce, with rank 0 entering last, and every step
spending ~3 ms in host prep plus 2-4 ms *inside* `cudaGraphLaunch`. That is the
profiler: CUPTI per-kernel activity makes a ~3,400-node graph launch cost
milliseconds. Measured instead under Nsight Systems with
`--cuda-graph-trace=graph` over one `cudaProfilerApi` window
(`NSYS_OUT=... serve.sh`, 46 steps x 4 ranks, `nsys_steps.py`):

| rank | period | verify graph | other GPU (drafter, sampling, prep) | GPU idle | host-late | cudaGraphLaunch p50 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 25.56 | 23.55 | 1.70 | 0.31 | 0.001 | 301 us |
| 1 | 25.56 | 23.54 | 1.67 | 0.34 | 0.000 | 292 us |
| 2 | 25.55 | 23.54 | 1.67 | 0.34 | 0.000 | 290 us |
| 3 | 25.56 | 23.55 | 1.69 | 0.31 | 0.000 | 280 us |

Graph-start skew across ranks: p50 18 us, p90 24 us (one 0.34 ms outlier in 47
steps). The step is GPU-bound: the step-entry stall (~0.02 ms, not 0.96) and
drafter launch starvation are profiler artifacts; there is no host-side lever.
`entry_stall.py` / `dive_launch.py` are the torch-trace probes that showed the
artifact.

## GEMM grids (trace 2108844)

The earlier dive's "grid 2-4" read only grid.x: the nvjet GEMMs launch 4x16,
4x22, 4x24 = 64-96 CTAs. The large projections are near the HBM floor
(o_proj 50 MB in 14.1 us ~ 3.6 TB/s; q_b and fused qkv_a ~82%); the dense-GEMM
headroom is in the small ones (W_UK/W_UV bmm, router, shared expert) and the
split-K reduce kernels, ~0.5 ms/step.

## Fused one-shot DCP ops (vLLM, uncommitted)

Per layer the DCP4 MLA decode ran `torch.cat(ql_nope, q_pe)` + one-shot
query all-gather, and LSE all-gather + `_correct_attn_cp_out_kernel` +
one-shot reduce-scatter. `one_shot.cu` gains two kernels on the same custom
all-reduce buffers:

* `all_gather_cat(a, b)`: each rank reads peers' `a` and `b` rows (strided:
  the transposed bmm output and the q slice) and writes the concatenated,
  head-gathered query; no concat kernel.
* `lse_reduce_scatter(out, lse)`: one barrier pair; each rank reads every
  peer's LSE and partial output for its heads, computes the global LSE
  (Triton's butterfly summation order, `ex2.approx` exp, libdevice log), scales,
  rounds each partial to bf16 and sums in rank order, exactly as the three
  kernels it replaces.

`VLLM_DCP_ONE_SHOT_FUSED` (default 1) gates both; either falls back to the old
path when unusable. `test_one_shot_fused.py` (torchrun, 4 GPUs, job 2111470):
query gather exact; combine **bitwise identical** to the production path in
6/6 rounds (eager and captured, with -inf / NaN / +inf LSEs). Captured, 78
calls:

| op | unfused | fused | per step (78 layers) |
|---|---:|---:|---:|
| query gather | 8.99 us | 7.90 us | -0.09 ms |
| attention combine | 11.15 us | 6.39 us | -0.37 ms |

Task-set A/B (fused vs `VLLM_DCP_ONE_SHOT_FUSED=0`, both arms on two nodes):
running.

Task-set A/B, interim (02:28; fused arm on 2111470, unfused on 2111453/2111454,
81 common requests, `paired.py`): fused - unfused = **-0.35 ms/step (95% CI
-0.74 .. -0.03)**, 123.2 -> 125.8 tok/s. The fused arm on 2111454
(rows-dcpfused-b) balances the nodes.

## Sparse-MLA / DCP glue (exact)

Per layer, around the FlashMLA sparse decode (per-layer sequence in
`dive/r0-w0-layers.txt`): a `torch.zeros` valid-count fill and a
`torch.full_like(-1)` output fill before the 16-tile atomic index remap; after
the kernel, `mask_empty_dcp_lse` (`valid_counts == 0`, `masked_fill`) and a
contiguous copy of the transposed `[H, T]` LSE.

* vLLM e1f28963b8: upstream #50365 (single-tile remap, no counter fill, no
  atomics) + #57458's in-kernel -1 tail, so no pre-fill.
  `test_index_remap.py`: same counts and valid sets as the frozen old kernel
  (`sparse_utils_ref.py`; the old prefix order depended on atomic arrival),
  -1 tails, exact non-DCP output; 3.53 -> 2.65 us per layer (graph, 78 calls).
  test_indexer_dcp_localize / test_flashmla_sparse_dcp: 39 passed.
* vLLM c85de9aa1e: FlashMLA sparse returns lse = +inf for an index row with
  no slot on the rank (`probe_empty_lse.py`, measured), and both
  `lse_reduce_scatter` and the Triton combine already drop +inf / NaN shards,
  so the mask cannot change the result: skipped when the layer's combine is
  `cp_lse_ag_out_rs` (not a2a); `lse_reduce_scatter` reads the transposed LSE
  in place. Bitwise identical to mask + copy + gather + correct + RS
  (`test_one_shot_fused.py`, 6/6); 15.11 -> 6.48 us per layer.
* Uncommitted: layers without their own indexer (57 of 78) reuse the
  DCP-converted indices of the last indexer layer (same buffer, keyed by
  token count / block table / rank). Eager equality check
  (`VLLM_DEBUG_CHECK_DCP_INDEX_REUSE=1`) queued on 2111720.

Task-set arms for e1f28963b8 + c85de9aa1e: rows-glue-{d,e} (2111654/2111655),
against rows-dcpfused-{b,c}.

## Result after the fixes (2026-09-29 evening, holds 2113264 / 2113265)

vLLM HEAD 911f3a2757 = a98ca6938c (fused DCP ops) + e1f28963b8 (single-tile
remap) + c85de9aa1e (in-place LSE, no mask) + b793523008 (reuse converted
indices on the 57 non-indexer layers; 10000/10000 in-model equality checks) +
911f3a2757 (padding dropped in route_prep, no mask_replica_padding).

Task set, paired (`paired.py`), 4 HEAD arms rows-head-{1..4}:

| comparison | common requests | step delta (95% CI) | step-weighted |
|---|---:|---:|---:|
| HEAD vs e1f+c85 (rows-glue-{d,e}) | 124 | -0.79 ms (-0.93 .. -0.64) | 25.11 -> 24.33 ms |
| HEAD vs unfused start (rows-dcpunf-{a,b}) | 122 | **-1.76 ms (-2.00 .. -1.50)** | 26.03 -> 24.29 ms |

(Earlier steps: fused vs unfused -0.22 ms (-0.45 .. -0.03), 108 requests;
e1f+c85 vs fused -0.65 ms (-0.87 .. -0.42).)

nsys, graph-level (`NSYS_OUT`, `nsys_steps.py`), 57 steps x 4 ranks:

| per step | baseline 2111454 | HEAD 2113264 |
|---|---:|---:|
| period | 25.56 ms | **23.73 ms** |
| verify graph | 23.55 | 21.72 |
| other GPU (drafter, sampling, prep) | 1.70 | 1.70 |
| GPU idle | 0.31 | 0.32 |

Torch-profiler dive of HEAD (traces/glm53-agentic-head-prof-2113265, 4 windows,
`dive/head-2113265.txt`) against 2108844 (`dive/r0-w0-layers.txt`; that trace
ran reserve 10): graph launches 3407 -> 2576 per step, 41 -> 30 kernels per
layer, in-graph idle 1.06 -> 0.74 ms. Gone per step: 252 int fills, 78 concats,
78 int->bool compares, 153 masked_fill / where, 78 _correct_attn_cp_out, 78
reduce-scatters, 156 of 177 gathers (78 gather_cat + 78 lse_reduce_scatter
instead), 57 of 78 index conversions.

## All-reduce + residual + RMSNorm fusion (FUSE_AR_RMS, FlashInfer)

The old `fuse_allreduce_rms` crash (illegal address) is upstream #48075: these
nodes have no NVLink multicast (`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` = 0
on all 4 GPUs, direct NV6 links, no NVSwitch; vLLM's symm-mem check logged the
same), yet `auto` picked FlashInfer's mnnvl backend, whose workspace is created
anyway and whose kernel then faults. vLLM b844ed93ba: `auto` asks the driver
and picks trtllm without multicast (checked on the node). With trtllm the
engine then aborted at start: the fusion splits the compile range at its token
limit, and DFlash2's compiled candidate selector (all-static inputs, num_reqs
1) got two compiled entries -> "Expected exactly one compiled range_entry";
vLLM 493acb5744 serves a fully static graph from any compiled entry.

Task set, arms rows-arrms-{1,2} (VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm,
FUSE_AR_RMS=true) vs rows-head-{1..4}, 129 common requests: **-0.15 ms/step
(95% CI -0.36 .. +0.00)**; accepted drafts/step 2.45 -> 2.56 (+0.11, se 0.05;
a rounding-order change, read as noise). Well short of the ~0.5 ms the removed
norm kernels suggest; not made default. Profile of the fused build:
traces/glm53-agentic-arrms-prof-2115784 (in progress).

Decode tok/s on the 84 requests common to start / HEAD / fusion arms, at a
common 3.45 tokens/step: 132.6 -> 142.3 -> 143.4.
