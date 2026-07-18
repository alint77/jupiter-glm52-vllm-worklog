# Speed-of-light re-analysis of the qualified MTP3-overlap configuration

This is an independent re-analysis of the fresh no-SP control trace
(`glm52-mtp3-nosp-control-profile-969649`), which profiles the current
qualified default: MTP3, verification overlap, no sequence parallelism,
exact-400K context. Unlike the earlier roofline (built on the serial MTP3
profile 968182), this measures the deployed configuration and adds
GPU-annotation-aligned phase segmentation, stream-concurrency depth, and
solo-time (critical-path) attribution.

Analyzers: `analyze.py` (component mapping, idle windows), `analyze2.py`
(concurrency scaffolding), `analyze3.py` (final phase/overlap/solo pass).
Run against decompressed rank traces with `.venv/bin/python analyze3.py`.

## Step anatomy

Six steady steps per rank, all four ranks within 0.01 ms of each other:

```text
step wall (in-profile)   28.615 ms   (real steady step ~22.7 ms: 123.65 tok/s
                                      at 60.25% acceptance = 2.81 tok/step)
  target verify window   26.50 ms    93% of the step (78 layers, q=4)
    union GPU busy       24.06 ms
    micro-idle           ~2.45 ms    thousands of <50 us launch gaps; no
                                     single idle window >= 50 us anywhere
  draft/sample gap        2.12 ms    3 MTP passes (~0.65 ms each) + sampling
```

The CPU launches ~17 ms ahead of GPU execution and its per-step work is
8-16 ms; the host is not a bottleneck. Profiling inflates the wall by ~26%;
ratios below are the reliable signal.

## Where the target window goes (rank mean, ms/step)

Busy sums count each stream separately; solo time is when that component is
the only thing running on the GPU (critical-path proxy).

| Component | Busy | Solo | Calls | Notes |
| --- | ---: | ---: | ---: | --- |
| Routed W4 Marlin (hot+cold) | 11.87 | 5.59 | 300 | union span 7.37; overlap saves 4.50 |
| Custom TP4 all-reduce | 4.45 | 4.45 | 166 | 100% solo: a hard sync point |
| Dense W4 Machete | 3.62 | 3.17 | 312 | ~11% of memory roof at M=4 |
| MoE support (sort/align/silu/sum) | 2.88 | 0.81 | 600 | per-tier-per-layer duplication |
| Elementwise/norm/metadata | 1.93 | 1.69 | ~1250 | Inductor + eager small launches |
| Sparse FP8 MLA | 1.87 | 1.58 | 160 | 16 real heads padded to 64 |
| DSA scan + top-k | 1.46 | 1.31 | 46 | context-proportional |
| Dense W4 Marlin (QKV-A) | 0.87 | 0.87 | 78 | |
| MLA BF16 contractions | 0.81 | — | ~200 | |

Concurrency depth during target busy time: 70% single-stream, 30%
two-stream, ~0% three-stream. The hot/cold Marlin overlap is real
(4.4-4.6 ms of Marlin||Marlin per step) but nothing else overlaps.

## Findings

1. **The all-reduce bar is desynchronization wait, not communication.**
   166 calls x ~28 us mean against a measured 4.06 us isolated floor
   (0.71 ms/step). Every one-stage reduction spins until the slowest rank
   arrives, so per-layer expert-load jitter is billed to the AR kernel.
   Cross-rank totals confirm it: rank 3 has the most routed work (12.18 ms)
   and the least AR time (4.21 ms); rank 0 the least routed (11.65) and the
   most AR (4.63). Total-level imbalance is only ~0.5 ms; the remaining
   ~3.7 ms is per-layer jitter and launch granularity.
2. **The cold Grace tier paces the routed span.** Hot-only weight streaming
   at 3.5 TB/s needs ~2.9 ms; the observed union span is 7.37 ms and the
   Marlin||Marlin overlap (~4.5 ms) matches a C2C-bound cold branch
   (0.8-1.9 GB at <=421 GB/s is 1.9-4.5 ms). The overlap already hides the
   hot tier behind the cold one; further gains need less cold traffic, not
   more streams. Placement is still optimized from pre-MTP single-token
   traces, while verification touches ~4x the experts per step.
3. **~4.8 ms/step is small-kernel tax.** MoE support + elementwise are
   ~1,850 launches/step; align/sort/count run per tier per layer (150x) for
   a 4-token batch. This also explains most of the 2.45 ms of sub-50 us
   micro-idle: ~3,600 kernels/step at ~0.7 us average gap.
4. **Sparse MLA still burns 4x head padding** (~1.4 ms/step recoverable
   with a native 16-head decode path, e.g. FlashMLA-ETAP).
5. **Machete dense W4 runs at ~5x above its byte floor** (2.45 GB/step
   needs 0.70 ms; observed 3.62 ms) — small-M inefficiency.
6. **The draft is already cheap (7% of the step); acceptance is the only
   draft-side lever.** Tokens/step = 1 + 3 x acceptance, so +5 pp
   acceptance is +5.3% throughput multiplicative with any step-time cut.
   This also quantifies the DFlash risk: with block size 16 the dominant
   93% target-verify cost grows (more unique experts per verify), so
   long-context acceptance must be far above MTP3 to net out positive.

## Speed of light

Per-rank target-window bytes: ~10.3 GB hot routed + 2.45 GB dense W4 +
~1.3 GB DSA scans + ~1 GB vocab/KV/misc = ~15 GB HBM (4.3 ms at 3.5 TB/s)
plus 0.8-1.9 GB cold C2C (1.9-4.5 ms at 421 GB/s, overlappable). Step SOL
including drafts is ~6-6.5 ms — about 440 tok/s at current acceptance. The
implementation is at ~3.5x SOL.

## Ranked opportunities (real-scale estimates per step of 22.7 ms)

| # | Change | Est. saving | Mechanism |
| --- | --- | ---: | --- |
| 1 | MTP-aware placement + per-layer load balance rebuild | 1.5-2.5 ms | shrinks both AR wait (jitter) and cold C2C traffic |
| 2 | Fuse MoE support chain; batch small launches | 1.5-2.0 ms | 1,850 launches -> few hundred; also shrinks micro-idle |
| 3 | Native 16-head sparse MLA decode (FlashMLA-ETAP) | ~1.1 ms | removes 75% padded head work |
| 4 | Machete/Marlin small-M tuning for GH200 | 0.5-2.0 ms | uncertain; needs kernel work |
| 5 | Draft-head acceptance (orthogonal) | +5.3%/5 pp | fine-tune or replace draft; DFlash risky at 400K |

Items 1-3 alone project ~17-18 ms real steps: ~155-165 tok/s at unchanged
acceptance, versus 123.65 today. Code touch points: placement/planner
(`tiered_moe_placement.py`, `tiered_moe_planner.py`), tiered dispatch
(`modular_kernel.py:apply_tiered`), sparse MLA kernel selection
(`vllm/models/deepseek_v32/nvidia/attention.py`), custom AR only insofar as
balance fixes reduce its wait share.

Rejecting local argmax was correct: vocabulary collectives are 0.06 ms/step
and the host has ~14 ms/step of slack; there was nothing to win.

## Per-kernel hardware analysis (`analyze4.py`, rank 0)

| Kernel family | n/step | p50/p90/p99 us | Grid | Achieved vs roof | Limiter |
| --- | ---: | --- | --- | --- | --- |
| Routed MoE Marlin | 300 | 32/77/126 | (396,1,1) fixed | ~1.0 TB/s blended, 29% | M=4 AI, C2C cold tail, ~20% near-empty calls (<=2.5 us) |
| Custom all-reduce | 166 | 6.5/93/162 | (6,1,1) | p50 near 4.06 us floor | tail-only: slow 20% of calls carry ~80% of time = desync wait |
| Dense W4 Machete | 312 | 13/17/25 | (1,132,1) uniform | 535 GB/s, 15% | M=4 tile shape |
| QKV-A W4 Marlin | 78 | 11.2/11.5/11.9 | (132,1,1) | 598 GB/s, 17% | M=4 |
| Sparse FP8 MLA | 160 | 12/16/16 | (1,4,33)+(1,4,8) | ~7% useful | 16 heads padded to 64 |
| DSA K scan | 24 | 45/46/47 | (132,1,1) | 1.25 TB/s, 36% | scan structure; context-bound |
| DSA top-k | 24 | 26/27/29 | (4,8,1) = 32 blocks | 24% SM utilization | cooperative grid too small for 132 SMs |
| MoE support | 600 | 3.3/7/26 | (1..32,1,1) | launch-bound | 1-2 block kernels |
| Elementwise/meta | ~1200 | 1.5/2.8/4.2 | (1..8,1,1) | launch-bound | graph node granularity |
| Vocabulary GEMM | 4 | 146/147/148 | (2,66,1) | 3.25 TB/s, 93% | at roof; leave alone |

The all-reduce distribution is the smoking gun for the jitter thesis: the
median call is already at the isolated floor; the p90+ tail after imbalanced
MoE sections is where 80% of AR time lives.

## Parallelism alternatives (analysis only; nothing measured)

TP2xPP2 was never tested; the tiered validator pins TP4/EP4. The MLA KV
intuition is correct: the compressed latent cache is replicated across TP
ranks, so TP4 stores the 19.06 GiB 400K cache four times, and TP2xPP2 would
cut per-GPU KV to ~9.5 GiB (39 layers), freeing ~490 hot-expert slots.
But at batch one PP stages execute serially: each GPU still streams the
same ~15 GB of weights/scans per step, and the two stages run one after the
other, halving aggregate HBM bandwidth utilization. Memory-bound components
(~20 of 26.5 in-profile ms) would roughly double per step against ~4-5 ms
of collective/padding/cold gains. Estimated net: ~1.5x slower (~75-85
tok/s). Rejected analytically; not worth a node-hour.

Decode context parallelism (dcp=4) is the right tool for the same win: KV
shards to ~4.8 GiB/GPU (freeing ~14.3 GiB = ~735 hot experts), and the
context-bound DSA scan and sparse MLA shard 4 ways, without stage
serialization. Cost: per-layer attention combine collectives, and the
active FlashMLA sparse backend has no DCP support - only
`flashinfer_mla_sparse` implements it (`flashmla_sparse.py` has no dcp
path; `indexer.py` and `sparse_utils.py` do). A DCP experiment therefore
requires a backend switch and MTP-under-DCP verification: medium risk,
real upside (~1.5-3 ms/step estimated).

## Concurrency 4 and the DCP-on-SM90 situation

KV per rank at 400K is 656 B/token/layer x 78 layers = 19.06 GiB per
request, replicated across TP4. Concurrency math:

| Config | KV/rank | Feasible? |
| --- | ---: | --- |
| c=1 x 400K, TP4 (today) | 19.06 GiB | yes (current) |
| c=4 x 400K, TP4 | 76.2 GiB | no - exceeds free HBM ~4x |
| c=4 x ~100K, TP4 | 19.06 GiB | yes with contract relaxation only |
| c=4 x 400K, TP4+DCP4 | 19.06 GiB | yes - same footprint as today |

No SM90 sparse backend currently supports DCP: `flashinfer_mla_sparse`
(DCP yes) is `capability.major == 10` only; `flash_attn_mla_sparse` is
SM90 but BF16-KV-only and explicitly rejects DCP; `flashmla_sparse`
(current, fp8_ds_mla) has no DCP path. Getting DCP means porting it into
`flashmla_sparse`. All building blocks are in-tree:

- `indexer.py` and `sparse_utils.py` are already DCP-aware (local row
  bounds, dcp_size-aware top-k index conversion).
- `cp_lse_ag_out_rs`/`_ar` LSE-merge helpers in
  `vllm/v1/attention/ops/common.py`, already used by `mla_attention.py`.
- `flashattn_mla` (dense, SM90) demonstrates the DCP head-folding pattern
  (`num_heads_q = num_heads * dcp_world_size`).
- `flashmla_sparse` already has `supports_spec_as_decode=True` for MTP.

Synergy: DCP4 folds 16 local heads x 4 ranks = 64 query heads, exactly
the FlashMLA sparse kernel's native head count. The current 75% padding
waste becomes useful work, so DCP4 subsumes the FlashMLA-ETAP item, and
the per-rank DSA scan/top-k shrink ~4x (context/4 per rank).

Batch-4 also amortizes weight streaming: 16 verify tokens x top-8 =~ 99
unique experts per layer globally (vs ~29 at c=1), so hot bytes rise
~3.4x while served tokens rise 4x; dense weights amortize exactly 4x.
Estimated aggregate at c=4 x 400K with DCP4: ~2.2-2.5x today's
throughput (~270-310 tok/s aggregate) at ~70-80 tok/s per request.

Plan: (0) c=4 at ~90-100K today with only tiered-contract relaxation
(max_num_seqs=4, graph size 16, overlap gate to 16) to validate batching;
(1) DCP4 port in flashmla_sparse, qualified at c=1 x 400K against the
exact-output SHA; (2) c=4 x 400K = DCP4 + relaxed contract. The tiered
validator currently pins TP4/DP1/PP1/PCP1/DCP1, max_num_seqs=1,
max_model_len=400000, fp8_ds_mla, block 64 (vllm/config/vllm.py); phases
0 and 2 require deliberate relaxation of those pins.
