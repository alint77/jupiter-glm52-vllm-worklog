# MTP3 c4 critical-path review and optimization plan

Independent re-analysis of the four rank traces in

```text
/e/scratch/profound/naeimitabiei1/glm52-c4-mtp3-edge-profile-1022402
```

Steady concurrency-four decode at 395,904 prefix tokens, MTP3, TP4/EP4/DCP4,
`ag_rs`, V2 runner, full CUDA graphs, 7 GB tiered reserve, 3,023 hot / 1,777
cold expert slots per rank. Companion to
[`c4-mtp3-sol-idle-analysis.md`](c4-mtp3-sol-idle-analysis.md); where the two
disagree, the disagreement is stated explicitly and the evidence given.

Target regime: this configuration is intended to host an autoresearch agent and
its subagents, so the figure of merit is **aggregate tokens/s across four
concurrent sequences**, not single-stream latency.

## Method differences from the earlier report

Three changes materially altered the conclusions.

1. **Step segmentation from CUDA-graph correlation IDs, not annotations.**
   Kernels replayed from a graph inherit the `cudaGraphLaunch` correlation ID,
   so each engine step decomposes exactly into its four graph replays plus
   eager work. One of the six steady steps (the fourth) has a *truncated*
   `gpu_user_annotation` — 10.6 ms instead of ~58 ms — while still containing a
   full 4,794-op workload. Annotation-derived spans silently drop it.
2. **Host-side API timeline included.** The earlier report attributed
   inter-graph GPU idle to host orchestration without checking host timing.
   The host actually runs ~50 ms *ahead* of the GPU for most of the step.
3. **Tier identification from source semantics.** In
   `modular_kernel.py:apply_tiered`, cold runs on `aux_stream()` and hot on the
   current stream; `cold_output.add_(hot_output)` executes on the *main* stream
   after the join. So the stream carrying the `CUDAFunctor_add` after the two
   `moe_sum_vec` calls is the hot tier. `compressed_tensors_moe_wna16_marlin.py`
   builds tiers in `("hot", "cold")` order, confirming `tiers[0]` is hot.
   All 75 routed layers per step resolve cleanly under this rule.

Five steady steps per rank, four ranks, 20 rank-steps unless stated.

## Step structure

Mean step wall **62.43 ms** (rank spread 62.42–62.46 ms; step-to-step 61.6–63.3
ms). 4,794 GPU operations per step in four graph replays plus ~120 eager ops.

| Region | Wall | GPU idle | Idle share |
| --- | ---: | ---: | ---: |
| Eager prologue (input prep) | 3.01 ms | 2.73 ms | **91%** |
| Target graph (4,436 nodes) | 55.86 ms | 3.78 ms | 6.8% |
| Target → draft 1 | 0.35 ms | 0.04 ms | 12% |
| Draft graphs 1–3 (89/85/70 nodes) | 2.12 ms | 0.11 ms | 5% |
| Draft 1→2 and 2→3 gaps | 1.57 ms | 1.53 ms | **97%** |
| **Total** | **62.43 ms** | **7.95 ms** | 12.7% |

The host issues the target graph and draft graph 1, then blocks in a single
`cudaStreamSynchronize` for **50.7 ms**. The engine is GPU-bound overall; the
host has ~50 ms of spare wall time per step. That reframes every idle region:
none of it is host *throughput*, all of it is host *round-trip serialization*.

## Critical-path budget

"Solo" is time when only that category is executing on the GPU — its direct
share of the critical path. Union is that category's own busy span.

| Category | cum | union | **solo** | share of step | n/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Routed W4 Marlin (MoE) | 42.31 | 25.27 | **20.15** | 32.3% | 300 |
| TP custom all-reduce | 9.27 | 9.27 | **9.27** | 14.8% | 166 |
| *GPU idle* | – | – | **7.95** | 12.7% | – |
| Dense/shared cutlass + nvjet | 5.02 | 5.02 | **4.94** | 7.9% | 502 |
| Glue (elementwise/triton/other) | 4.92 | 4.82 | **4.65** | 7.4% | 2,410 |
| DCP NCCL AllGather/ReduceScatter | 3.12 | 3.12 | **3.12** | 5.0% | 271 |
| Sparse MLA (FlashMLA) | 2.65 | 2.37 | **2.37** | 3.8% | 162 |
| DSA indexer + top-k select | 1.89 | 1.89 | **1.89** | 3.0% | 72 |
| MoE epilogue (act/sum/align/topk) | 6.61 | 6.29 | **1.17** | 1.9% | 678 |
| Dense W4 Marlin | 0.92 | 0.92 | **0.92** | 1.5% | 78 |
| MTP FP8 blocks | 0.58 | 0.56 | **0.51** | 0.8% | 48 |
| KV cache write | 0.21 | 0.21 | **0.21** | 0.3% | 105 |

All 400K-context-specific work — sparse MLA, DSA indexer, top-k, KV write, DCP
collectives — totals **7.6 ms, 12% of the step**. Context length is not what
limits this regime. Routed MoE and its synchronization are 47% of it.

## Finding 1 — the entire TP all-reduce cost is MoE expert skew

Classifying each of the 166 custom reductions by its producer:

| Position | n/step | ms/step | mean | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| post-attention | 86.3 | **0.61** | 7.01 us | 6.08 | 7.58 | 102 us |
| post-MoE | 76.0 | **8.20** | 107.9 us | 86.9 | 261 | 591 us |
| unclassified | 3.5 | 0.46 | 131 us | 72 | 480 | 865 us |

The post-attention reductions sit at the 5.0 us payload floor with a p90 of
7.6 us: **attention is essentially perfectly balanced across ranks** — DCP4
shards the 400K context evenly and it works. Post-MoE reductions carry
**7.82 ms/step of wait above the same floor**.

An independent measurement agrees. Summing, over the 75 routed layers, the
per-layer excess of the slowest rank's MoE span over the four-rank mean gives
**7.76 ms/step** — within 0.8% of the 7.82 ms of all-reduce wait, derived from
completely different quantities. The all-reduce is a barrier faithfully
reporting expert-load skew, and nothing else.

**New decomposition:** how much of that skew is statically fixable?

| Component | ms/step | share |
| --- | ---: | ---: |
| Persistent per-layer rank ordering | **5.96** | 77% |
| Step-to-step routing variation | 1.80 | 23% |

The skew is *diffuse*, not concentrated: the 15 worst layers hold only 35% of
it, the rest is ~0.08 ms spread over every layer. No single rank is globally
slow — rank 3 is slowest in 25 layers, rank 0 in 18, rank 2 in 17, rank 1 in 15,
and total per-rank routed time spreads only 5.0% (1.28 ms). This is a
**per-layer** ownership problem, and 77% of it is a fixed property of the
current owner map rather than run-to-run noise.

## Finding 2 — the cold Grace tier is at the C2C roofline and costs half what it appears to

All 300 Marlin MoE launches per step use an identical geometry: grid
`(396,1,1)`, block `(128,1,1)`, 76,458 B shared memory, 3.0 blocks/SM.
GH200 has 132 SMs, and 3 × 76,458 = 229,374 B against 233,472 B per SM. So
**every Marlin launch requests exactly one full-occupancy wave of the whole
GPU**, and does so identically at 8, 12 and 16 tokens. Two tiers launched
back-to-back therefore cannot co-reside at launch; the second tier's blocks
only start as the first tier's retire.

That distorts the measured durations, and the distortion is visible as a
physical inconsistency. Per expert, w13 is 12.976 MB and w2 is 6.488 MB —
exactly 2:1. Measured per-step time ratios:

| Tier | w13 | act | w2 | w13/w2 |
| --- | ---: | ---: | ---: | ---: |
| Hot | 16.67 ms | 0.51 | 7.34 ms | **2.27** |
| Cold | 6.58 ms | 2.60 | 11.72 ms | **0.56** |

Hot's 2.27 tracks the 2.0 byte ratio — hot is cleanly weight-bandwidth-bound.
Cold's 0.56 is unphysical and stable across all three MTP depths (0.55–0.57),
so it is systematic, not noise. Cold w13 is the one launch that starts
uncontended; cold w2 and cold `act_and_mul` execute while hot w13 holds the
SMs. Reading cold w13 as the true rate:

- 6.58 ms of cold w13 at 421 GB/s ⇒ **2.85 activated cold experts per layer**
- cold w2 should then be 3.29 ms; it measures 11.72 ms — **3.6× dilation**
- **cold solo cost ≈ 10.4 ms/step, not the 20.9 ms it appears to be**

Cross-check: 2.85 activated of 23.7 cold slots per layer (12%) against roughly
19–22 activated of 40.3 hot slots (~50%) is exactly the profile a
routing-frequency-ordered placement should produce. The numbers are consistent.

The hot tier then carries ~19–22 activated experts per layer at
**1.0–1.3 TB/s effective, 29–38% of HBM peak** — and the hot chain (24.52 ms)
is within **1.19 ms** of the observed layer-span total (25.72 ms). Overlap is
already near-perfect; the hot tier alone is the routed critical path.

This corrects the earlier report's framing. Its conclusion that removing cold
work saves only ~1.1 ms is right, but the reason is not that cold is a small
amount of work running alongside a dominant hot path — it is that cold is only
~10 ms of work hiding inside a 24.5 ms hot window, with **~14 ms of unused
Grace bandwidth per step**.

## Finding 3 — the residency planner is pushing experts the wrong way

Per activated expert per step across all 75 layers:

- hot: **1.09–1.43 ms** (depending on the true distinct-expert count)
- cold: **3.47 ms** (fixed by the C2C roof, measured)

Cold is only 2.4–3.2× more expensive than hot, not the ~8.3× that the raw
bandwidth ratio (3.5 TB/s ÷ 421 GB/s) suggests — because the hot Marlin path
only reaches ~a third of HBM peak. With overlap active, the span is
`max(t_hot, t_cold)`, minimized when the two are equal. Currently
t_hot = 24.5 ms and t_cold = 10.4 ms — far off balance in the direction of
*too much in HBM*.

| Distinct activated experts/layer | current span | balanced span | saving |
| --- | ---: | ---: | ---: |
| 20 | 24.52 ms | 20.24 ms | **4.28 ms (6.9% of step)** |
| 22 | 24.52 ms | 20.57 ms | **3.95 ms (6.3%)** |
| 25.3 (uniform-routing bound) | 24.52 ms | 21.01 ms | **3.51 ms (5.6%)** |

Moving activation mass from HBM to Grace is worth **3.5–4.3 ms/step**, and it
frees HBM that can go to KV — i.e. to concurrency. The project has been moving
experts the other way (Phase 15 promoted hot 2,870 → 3,713 as budget grew; this
sweep runs 3,023 hot at the 7 GB reserve). **At c=4 with overlap enabled that
direction is backwards.**

Two caveats. The planner must balance *expected activation mass*, not slot
counts — 1,777 cold slots already yield only 2.85 activated experts per layer,
so the frequency profile, not the slot count, is the control variable. And the
optimum shifts if hot Marlin efficiency improves (Finding 5); the two must be
re-tuned together.

## Finding 4 — 4.3 ms/step of pure host round-trip, at ~95% GPU idle

| Region | wall | GPU busy | host API calls |
| --- | ---: | ---: | ---: |
| Eager prologue | 3.01 ms | 0.28 ms | ~50 launches, 20 H2D memcpys |
| Draft 1 → draft 2 | 0.83 ms | 0.026 ms | 21 |
| Draft 2 → draft 3 | 0.75 ms | 0.023 ms | 22 |

The prologue runs ~50 individually launched eager kernels — `_expand_idx_mapping`,
`_prepare_pos_seq_lens`, `_dcp_local_seq_lens`, `_combine_sampled_and_draft_tokens`,
`_gather_block_tables`, `_compute_slot_mappings`, `_prepare_uniform_decode` —
each preceded by 87–146 us of GPU idle, plus 20 H2D pinned copies averaging
52 us of idle each. Total GPU work: 0.28 ms. Total wall: 3.01 ms.

Between draft graphs the host wakes after the previous graph completes, issues
~11 eager kernels and 3 async copies, then launches the next graph. The GPU does
25 us of work per 0.8 ms gap.

Together **4.3 ms/step (6.9%)** is host serialization with the GPU essentially
empty — and the host has 50 ms of idle wall time available in the same step.
This is the cleanest recoverable time in the whole trace.

## Finding 5 — the in-graph idle is node count, and the glue is bigger than it looks

The 3.78 ms of idle inside the target graph is 3,326 gaps averaging 1.14 us,
none above 246 us. Attributing each gap to the kernel it precedes:

| Successor | idle ms/step | gaps/step | avg |
| --- | ---: | ---: | ---: |
| elementwise | 2.28 | 762 | 3.0 us |
| cutlass GEMM | 1.00 | 302 | 3.3 us |
| triton | 0.68 | 661 | 1.0 us |
| Memset | 0.50 | 179 | 2.8 us |
| NCCL AllGather | 0.73 | 190 | 3.9 us |
| **cross_device_reduce** | **0.10** | 166 | **0.6 us** |

Two things follow. The all-reduce is *never* preceded by idle — its wait is
inside the kernel, spinning on SMs, which is why it shows as 100% solo in the
budget. And the in-graph idle is entirely a small-node scheduling tax: 2,410
glue kernels cost 4.65 ms of solo execution *and* induce ~3 ms of dependency
gaps. Combined, small-kernel overhead is **~7.6 ms/step, 12% of the step** —
comparable to the all-reduce and larger than everything except routed Marlin.

This agrees with the earlier report's conclusion that these are graph-node gaps
rather than CPU launch latency, and the host timeline now proves it: the host
is 50 ms ahead and issues nothing during the target graph.

## Finding 6 — routed MoE scales almost linearly with batch, and that is the scaling ceiling

Same weights, same placement, same context; MTP1/2/3 give verify batches of 8,
12 and 16 tokens across four sequences.

| Verify tokens | Marlin cum | layer-span total | hot chain | cold chain |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 28.17 ms | 17.26 ms | 16.48 | 13.74 |
| 12 | 36.48 ms | 22.14 ms | 21.19 | 17.85 |
| 16 | 42.31 ms | 25.72 ms | 24.52 | 20.90 |

Linear fit on the span: **9.02 ms fixed + 1.057 ms per token** — only **35% of
the routed MoE cost is fixed weight streaming**. For a decode MoE this is the
wrong regime: with 64 local experts per rank and ~2 assignments per token per
rank, the activated-expert set is nowhere near saturation, so each added token
pulls in nearly its own fresh set of expert weights.

**This is the reason c=4 yields ~1.35× the c=1 aggregate rather than 4×**, and
it is more fundamental than any of the fixable items above.

The marginal cost is, however, *declining*: 1.221 ms/token from 8→12, 0.894
ms/token from 12→16. Amortization has begun. Extrapolating, c=8 would cost
meaningfully less than 2× the MoE time of c=4. Caveat: these points vary
*positions within a sequence*, whereas raising concurrency adds *independent*
sequences whose routing is less correlated, so the true concurrency slope is
probably somewhat worse. That needs a direct measurement, not an extrapolation.

## What is not worth pursuing

- **Cold-path elimination.** Removing all cold work shortens the routed span by
  ~1.2 ms. The right move is the opposite (Finding 3).
- **All-reduce as a bandwidth problem.** The payload floor is 5.0 us and the
  measured link utilization is a consequence of skew, not a cause. Fix expert
  balance and remeasure before touching the collective.
- **DCP collectives and sparse MLA.** 3.12 ms and 2.37 ms, both already near
  their floors, and attention is provably balanced. Note this contradicts the
  ~12 ms/step DCP figure quoted in the project README — measured here at
  3.12 ms across 271 calls.
- **Context-length work generally.** 12% of the step at 396K. Shorter agent
  contexts will not speed this configuration up much; more concurrency will.
- **The MTP head.** 0.51 ms/step. Leave it alone.
- **Deeper MTP.** Already settled at depth 3 by the sweep; nothing in the trace
  argues otherwise.

## Plan

Ordered by expected gain × confidence ÷ effort. Gains are per-step reductions
against the 62.43 ms baseline; they are not fully additive, since P1 and P2 both
act on the routed-MoE region.

### P1 — Per-layer EP load balancing (≈5.9 ms, 9.5%)

The Phase 6 machinery already supports a fingerprinted arbitrary per-layer EP4
owner map, validated and loadable in the full graphed server. Its objective was
"minimize cold routing". Add a second term: **minimize, per layer, the maximum
over ranks of expected activation mass**, using the existing domain-trained
per-expert frequency profile from `optimize_routing_profile.py`.

- Recoverable: the 5.96 ms persistent component. The 1.80 ms step-to-step
  component is out of reach for a static map.
- Gate: golden 400K SHA `d594e4d4…dcfc528` plus the 24-prompt realistic suite.
- Risk: low. Ownership changes are already exercised end-to-end.
- Verify: post-MoE all-reduce wait should fall from 7.82 ms toward ~2 ms; re-run
  the per-layer max-minus-mean measurement, which should track it.

### P2 — Rebalance activation mass toward Grace (≈3.5–4.3 ms, 5.6–6.9%)

Change the residency planner's objective from "fill HBM with the most frequent
experts" to "equalize expected per-tier execution time", i.e. drive
`t_hot ≈ t_cold` using the measured per-activated-expert costs (hot 1.09–1.43 ms,
cold 3.47 ms per step across all layers).

- Do P2 *after* P1, and re-fit the per-expert costs from the P1 trace.
- Secondary payoff: releases HBM for KV, feeding P6.
- Risk: medium. As cold mass grows, cold/hot SM contention grows; the current
  1.19 ms overlap gap could widen. Measure the layer-span total, not the
  per-kernel durations, which are contention-distorted.
- Cheap first probe: rerun this sweep's MTP3 point at 9 GB and 11 GB reserve
  (fewer hot slots) and check whether the realistic-suite aggregate *improves*.
  If the curve is non-monotonic in the direction predicted, the model holds.

### P3 — Remove the host round-trips (≈3.5–4.3 ms, 5.6–6.9%)

Two independent pieces.

- **Draft loop (1.57 ms).** Capture the three MTP proposer passes into one graph,
  or keep the loop resident on-GPU so draft *k*+1's input is built by device
  kernels rather than a host wake-up. The per-pass eager work is ~11 kernels and
  3 copies.
- **Prologue (2.73 ms).** ~50 eager launches and 20 H2D copies with 0.28 ms of
  GPU work. Either fold the input-prep kernels into the target graph's prologue
  region, batch the 20 H2D copies into one staged transfer, or enable async
  scheduling so prep for step *n*+1 overlaps step *n*'s 50 ms of GPU execution.
  The host demonstrably has the wall time free.
- Risk: low–medium; graph capture under DCP has bitten this project twice
  (`0d87dd9ae`, `67e6d48ff`), so expect capture-path debugging.

### P4 — Alternative W4A16 MoE kernel at M=16 (0 to ~9 ms, high variance)

Hot Marlin runs at 29–38% of HBM peak with a grid fixed at exactly one
occupancy wave regardless of token count — a shape tuned for large-M GEMMs.
`WNA16MoEBackend.FLASHINFER_TRTLLM` already exists in the tree as an
alternative. This is a cheap A/B with a potentially large payoff and an equally
real chance of no gain.

- Run as a standalone kernel benchmark first, under
  `benchmarks/kernels/`, at M ∈ {8,12,16,32}, not in the server.
- If a backend wins, P2's balance point moves and must be re-fit.

### P5 — Reduce graph node count (≈2–3 ms of ~7.6 ms, higher effort)

2,410 glue kernels per step cost 4.65 ms solo plus ~3 ms of dependency gaps.
Highest-density targets: the MoE epilogue (678 calls/step of `act_and_mul`,
`moe_sum_vec`, `moe_align_block_size`, `count_and_sort_expert_tokens`) and the
1,431 elementwise + 644 triton nodes. Fusing the two `moe_align`/`count_and_sort`
pairs across tiers, and the per-tier `moe_sum_vec` + `add_` into one kernel, is
the obvious first slice. Do this last: it is the most invasive and the gain per
unit of work is the lowest.

### P6 — Concurrency headroom for the agent workload

The routed-MoE marginal cost per token is declining, so c=8 should cost well
under 2× the c=4 MoE time. KV is the binding constraint: c=4 × 400K currently
uses 21.9 GB/rank of a ~96 GB budget against 64.4 GB of weights.

- P2 directly funds this by moving weights off HBM.
- Autoresearch subagents will mostly run far below 400K. Measure the aggregate
  throughput surface over (concurrency, context) rather than optimizing the
  400K corner — the 400K synthetic case is a memory and correctness stress
  test, and the realistic suite is already the primary gate (Phase 17).
- Required new measurement: an 8-sequence profile to get the *concurrency*
  slope, which the MTP-depth sweep cannot provide.

### Expected outcome

P1 + P2 + P3 land on independent parts of the step and are jointly worth
~13.8 ms if fully realized, i.e. 62.4 → ~48.6 ms (+28% aggregate throughput,
roughly 228 → 290 tok/s effective on the realistic suite). A realistic
partial-capture estimate of 60–70% of that is **+17–20%**. P4 and P5 are
upside on top; P6 changes the operating point rather than the step time.

## Suggested experiment order

Each entry is one Booster node for four hours; entries at the same level are
independent and should be submitted in parallel.

1. **Parallel, no source changes.** (a) MTP3 at 9 GB and 11 GB reserve to test
   the P2 direction empirically; (b) an 8-sequence c8 profile at a moderate
   context to get the concurrency slope; (c) a standalone `benchmarks/kernels/`
   W4A16 MoE backend comparison at M ∈ {8,12,16,32} on the login-node GH200.
2. **P1** — rebalanced per-layer owner map, gated on the golden SHA and the
   24-prompt suite, with a re-run of the post-MoE all-reduce measurement.
3. **P2** — re-fit tier costs from the P1 trace, retune the residency planner,
   remeasure.
4. **P3** — draft-loop graph capture and prologue batching, in that order.
5. Re-profile and re-derive this budget before starting P4 or P5.

Analysis scripts used for this report are reproducible from the traces alone;
the segmentation rule (graph correlation IDs) and the tier rule (`add_` stream)
are the only two non-obvious pieces.
