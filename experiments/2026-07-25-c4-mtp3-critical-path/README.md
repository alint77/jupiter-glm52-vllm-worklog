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

The skew is *diffuse*, not concentrated: the 15 worst layers hold only 35% of
it, the rest is ~0.08 ms spread over every layer. No single rank is globally
slow — rank 3 is slowest in 25 layers, rank 0 in 18, rank 2 in 17, rank 1 in 15,
and total per-rank routed time spreads only 5.0% (1.28 ms).

**How much of the skew is statically fixable? Almost none.** A first pass at
this measured per-rank means over the five steady steps and attributed
5.96 ms/step (77%) to "persistent per-layer rank ordering". That was wrong, and
the error was small-sample bias: with only five samples per rank, the maximum of
four noisy means overshoots the mean of means badly. Replaying the captured
routing traces through a validated cost model (`analysis/p1_decompose.py`) shows
the same estimator converging as the sample grows:

| Steps used | Apparent "persistent" component |
| ---: | ---: |
| 5 | 2.004 ms |
| 20 | 1.106 ms |
| 100 | 0.724 ms |
| 400 | **0.613 ms** |

The converged decomposition of the modelled 4.53 ms/step of skew is:

| Component | ms/step | share |
| --- | ---: | ---: |
| Expectation (reachable by a static owner map) | **0.61** | 14% |
| Per-step variance (order statistic) | 3.91 | 86% |

The shipped `greedy-balanced-owner` map is already near-optimal in expectation.
The skew is overwhelmingly the max-of-four-ranks order statistic on *which*
experts happen to fire in a given step, and no fixed assignment removes it.
Rebuilding the owner map against expected distinct-activation cost made things
slightly **worse** (+0.29 ms realized) and broke the per-rank hot-slot balance
that the HBM budget depends on ([2907, 2868, 2847, 2858] instead of 2870 each).

What actually creates the cost is **barrier granularity**. Summing the
per-layer maximum over ranks is far more expensive than taking the maximum of
per-rank totals:

| Quantity | ms/step |
| --- | ---: |
| `sum_layers max_rank` — 75 barriers, what we pay | 25.96 |
| `max_rank sum_layers` — 1 barrier, hypothetical | 22.01 |
| mean rank total | 21.43 |

**3.95 ms/step is the price of synchronizing 75 times on a quantity with
per-layer variance**, not of assigning experts badly. The layer-to-layer data
dependency is real, so the barriers cannot simply be removed; the lever is to
shrink the per-expert cost that the variance multiplies (see Finding 3 and P1).

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

> **Confirmed, after one false alarm.** A first isolated measurement appeared to
> show Grace-resident Marlin at 69–199 GB/s, and this derivation was briefly
> retracted. That measurement was wrong — its benchmark gave the cold tier's
> experts the whole routing, so it timed re-streamed blocks and divided by
> logical bytes.
> [2026-07-25-grace-bandwidth](../2026-07-25-grace-bandwidth/README.md) measures
> the cold path at **88–95% of the 421 GB/s roof** with production routing, so
> the roof assumption holds and ~2.85 (≈3.2 after the 0.9 factor) activated cold
> experts per layer stands. The isolated hot kernel does reach 61–62% of HBM
> peak, so the 29–38% figure below is too pessimistic: it charges cross-stream
> contention to the kernel.

Cross-check: 2.85 activated of 23.7 cold slots per layer (12%) against roughly
19–22 activated of 40.3 hot slots (~50%) is exactly the profile a
routing-frequency-ordered placement should produce. The numbers are consistent.

The hot tier then carries ~19–22 activated experts per layer at
**1.0–1.3 TB/s effective in situ**, against 2.14–2.18 TB/s measured in
isolation — so most of that gap is contention, not kernel quality.

> **Correction.** This section originally read the hot chain (24.52 ms) being
> within 1.19 ms of the layer-span total (25.72 ms) as evidence that overlap is
> "already near-perfect". It is not: both quantities are dilated by the same
> contention, so comparing them proves nothing. Isolated per-layer costs give an
> ideal overlapped span of **13.2 ms** and a serial span of 25.1 ms; the measured
> 25.7 ms matches *serial*. The tiers capture only ~25% of the available overlap,
> leaving ~12 ms/step on the table. See
> [2026-07-25-grace-bandwidth](../2026-07-25-grace-bandwidth/README.md). The same
> error invalidates the "39–40% overlap saving" quoted here and in the earlier
> SOL analysis, both of which compare against a sum of dilated durations.

This corrects the earlier report's framing. Its conclusion that removing cold
work saves only ~1.1 ms is right, but the reason is not that cold is a small
amount of work running alongside a dominant hot path — it is that cold is only
~10 ms of work hiding inside a 24.5 ms hot window.

That leaves ~14 ms/step of *aggregate* Grace bandwidth unused, which looks like
an invitation to move more experts there. Finding 3 shows it is not: the idle
C2C capacity is spread thinly across cells that do not need it, while a quarter
of cells are already cold-bound. Aggregate slack and per-layer slack are
different quantities, and only the latter sets the span.

## Finding 3 — the tier frontier is nearly flat, so HBM should buy KV, not experts

Per activated expert per step across all 75 layers: hot **1.28 ms**, cold
**3.47 ms** (the latter fixed by the C2C roof). Cold is only ~2.7× more
expensive than hot, not the 8.3× the raw bandwidth ratio suggests, because hot
Marlin reaches only ~a third of HBM peak.

That aggregate ratio invites an obvious inference: t_hot = 24.5 ms against
t_cold = 10.4 ms looks badly off balance, so moving activation mass to Grace
should shrink the span by 3.5–4.3 ms. **That inference is wrong**, for exactly
the reason Finding 1 gives: the span is `sum_layers max_rank`, and balancing
aggregates does not balance per-layer maxima. Sweeping the hot-slot budget
through the validated model (`analysis/p2_tier_sweep.py`, span predicted to
within 1% of the measured 25.72 ms) gives a monotone result in the *opposite*
direction:

| Hot slots/rank | HBM/rank | modelled span | vs shipped | cold-bound cells | cold excess |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,300 | 64.2 GB | 26.34 ms | −0.93 | 11.3% | 2.19 ms |
| 3,023 *(actual run)* | 58.8 GB | 26.65 ms | −0.62 | 24.8% | 6.29 ms |
| 2,870 *(profile)* | 55.9 GB | 27.27 ms | — | 34.1% | 9.84 ms |
| 2,600 | 50.6 GB | 29.54 ms | +2.27 | 51.3% | 19.49 ms |
| 2,300 | 44.8 GB | 32.96 ms | +6.99 | 67.8% | 34.37 ms |

At the real operating point **24.8% of (layer, rank) cells are already
cold-bound**, and in those cells the Grace tier overruns the hot path by
6.29 ms/step in total. The aggregate "cold is only 10.4 ms" view hides this
completely: cold sits idle in three cells out of four and is badly over budget
in the fourth. Every expert moved to Grace has a ~1-in-4 chance of landing where
it becomes the binding constraint at 2.7× the cost.

So the project's existing direction — promote cold experts into HBM as budget
allows — is **correct**, and the existing `optimize_hot_tail_aware` objective is
the right shape.

The useful finding is the *shape* of the frontier, not its sign. Marginal value
per hot slot collapses:

- 2,870 → 3,023 slots: 0.617 ms for 153 slots = **4.0 us/slot**
- 3,023 → 3,300 slots: 0.310 ms for 277 slots = **1.1 us/slot**

At the current point one more hot slot costs 19.46 MB of HBM and buys 1.1 us,
or 0.002% of the step. The same 5.4 GB/rank spent on KV instead is roughly one
more concurrent 400K sequence — about +25% aggregate throughput for a 0.31 ms
(0.5%) step cost. **Past ~3,000 hot slots per rank, HBM should go to KV.**

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

- **Rebuilding the EP owner map.** Refuted offline (Finding 1). The shipped
  `greedy-balanced-owner` map leaves ≤0.61 ms/step of expectation imbalance;
  86% of the skew is an irreducible order statistic. A rebuild against expected
  activation cost measured *worse*.
- **Moving activation mass to Grace.** Refuted offline (Finding 3). The span is
  monotone in the wrong direction and 24.8% of cells are already cold-bound.
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

Two of the three items this report originally proposed were refuted by offline
replay of the captured routing traces before any cluster time was spent; see
"What is not worth pursuing". The revised ordering is below. Gains are per-step
reductions against the 62.43 ms baseline and are not fully additive.

### P1 — A W4A16 MoE kernel that suits M=16 — **REFUTED, see below**

> Measured in
> [2026-07-25-marlin-decode-tuning](../2026-07-25-marlin-decode-tuning/README.md).
> Sweeping `blocks_per_sm` and the thread config changes nothing (every value
> within 0.4% of the auto heuristic; explicit thread configs 13-27% worse), and
> splitting the SM budget across the two tiers gains 0.0-0.4%. The isolated hot
> kernel already reaches 53-59% of HBM peak, so there is no large kernel win to
> collect. What the experiment did establish is that in-situ hot costs 1.48x its
> isolated time - about 7.8 ms/step of cross-stream contention - and that the
> cold tier's cost grows 2.89x from M=8 to M=32 while the hot tier's is flat.
> **P2 below is now the leading item.** The reasoning that motivated P1 is kept
> for the record:

Hot Marlin runs at 29–38% of HBM peak, with a grid fixed at exactly one
occupancy wave (396 blocks = 3 × 132 SMs, shared-memory limited) *regardless of
token count* — a shape tuned for large-M GEMMs, not a 16-row decode batch.

This is now the top item because it pays on two lines at once. Scaling the
modelled per-expert cost shows the skew shrinking in proportion, since the
order statistic multiplies the per-expert cost:

| Per-expert kernel cost | routed span | rank skew |
| ---: | ---: | ---: |
| ×1.00 (today) | 25.96 ms | 4.53 ms |
| ×0.75 | 19.47 ms | 3.39 ms |
| ×0.50 | 12.98 ms | 2.26 ms |

A 25% kernel improvement is worth ~6.5 ms of Marlin *plus* ~1.1 ms of
all-reduce wait — roughly 12% of the step. Nothing else in the trace has that
leverage, and it is the only item that attacks the 3.95 ms barrier-granularity
cost without touching the layer dependency structure.

- `WNA16MoEBackend.FLASHINFER_TRTLLM` already exists in the tree as an
  alternative to Marlin; the config plumbing is in place.
- Start with a standalone `benchmarks/kernels/` comparison at M ∈ {8,12,16,32}
  on the login-node GH200 — no Booster allocation needed, no server involved.
- Honest risk: this is a kernel-selection bet. It may return nothing.

### P1b — Make the hot and cold tiers actually overlap (≈12 ms, 19%) — **new leading item**

Measured in
[2026-07-25-grace-bandwidth](../2026-07-25-grace-bandwidth/README.md): isolated
at production routing, hot w13 is 115.3 us and cold w13 104.4 us, but their
union is 194.0 us against an ideal of 115.3 — only ~25% of the available
overlap is captured, and the trace's 25.7 ms layer-span total matches the
*serial* estimate (25.1 ms) rather than the ideal (13.2 ms).

This is the largest lever found anywhere in the analysis, and unlike the three
refuted hypotheses it rests on direct measurement of the isolated and combined
cases rather than inference from a single trace.

`blocks_per_sm` is already ruled out. Next probes, cheapest first:

1. Launch cold with a grid well below one wave — `sms * blocks_per_sm` never
   goes under 132 blocks, so the sweep never tested a genuinely small grid.
2. Lower-priority cold stream, so the scheduler backfills hot blocks.
3. Profile the isolated two-stream case to check whether the kernels are
   co-resident at all. If CUDA serialises them outright, the fix is structural
   rather than an occupancy tweak.

### P2 — Remove the host round-trips — **DONE, 3.94 ms/step (6.3%)**

> Root-caused and fixed in
> [2026-07-25-draft-sync](../2026-07-25-draft-sync/README.md). A single upstream
> line, `rank_offsets = torch.tensor(dcp_rank, device=...)` in
> `get_dcp_local_seq_lens`, materialized a Python int as a 0-d device tensor —
> a pageable H2D copy that blocks the host until the stream drains, executed
> once per draft step via `_build_draft_attn_metadata`. Replacing it with the
> Python scalar moved the host from launching just-in-time (37.5 us lead) to
> running a step and a half ahead (104.9 ms lead). Inter-graph GPU gaps fell
> from 4,547 to 607 us/step, which covers **both** halves below: the
> draft-to-draft gaps collapsed 786 → 70 us and the gap containing the eager
> prologue collapsed 2,617 → 173 us. Measured +3.1% to +7.0% output throughput
> with the golden SHA reproduced. Original scoping kept for the record:

Unaffected by the refutations; measured directly and mechanically clear.

- **Draft loop (1.57 ms).** Capture the three MTP proposer passes into one
  graph, or keep the loop resident on-GPU so draft *k*+1's input is built by
  device kernels. Today the host wakes after each draft graph, issues ~11 eager
  kernels and 3 copies, and relaunches — 0.8 ms of wall for 25 us of GPU work.
- **Prologue (2.73 ms).** ~50 eager launches and 20 H2D copies produce 0.28 ms
  of GPU work in 3.01 ms of wall. Fold the input-prep kernels into the graph,
  batch the H2D copies, or enable async scheduling so prep for step *n*+1
  overlaps step *n*. The host demonstrably has 50 ms/step free.
- Risk: low–medium. Graph capture under DCP has bitten this project twice
  (`0d87dd9ae`, `67e6d48ff`); expect capture-path debugging.

### P3 — Spend HBM on KV, not on hot experts

Finding 3 shows the hot-slot frontier is nearly flat past ~3,000 slots/rank:
1.1 us of span per 19.46 MB slot, or 0.002% of the step. The same HBM spent on
KV buys concurrency, and concurrency is what this deployment is for.

- Do **not** lower the hot-slot count (that direction is strongly negative).
  Simply stop raising it, and route any HBM freed elsewhere into KV.
- Trading 5.4 GB/rank of hot experts for KV costs 0.31 ms/step (0.5%) and buys
  roughly one more concurrent 400K sequence (~+25% aggregate).
- Needs a measured concurrency slope first: the 1.057 ms/token figure comes
  from varying positions *within* a sequence, and independent sequences will
  route less coherently. An 8-sequence profile settles it.

### P4 — Reduce graph node count (≈2–3 ms of ~7.6 ms, higher effort)

2,410 glue kernels per step cost 4.65 ms solo plus ~3 ms of dependency gaps.
Densest targets: the MoE epilogue (678 calls/step of `act_and_mul`,
`moe_sum_vec`, `moe_align_block_size`, `count_and_sort_expert_tokens`) and the
1,431 elementwise + 644 triton nodes. Fusing the per-tier `moe_align` /
`count_and_sort` pairs, and `moe_sum_vec` + `add_`, is the obvious first slice.
Most invasive, lowest gain per unit of work — do it last.

### Expected outcome

P2 and P4 are mechanical and jointly worth ~6–7 ms (62.4 → ~56 ms, +11%). P1 is
the only item that can move the step substantially — 12% for a 25% kernel win —
but it is a bet. P3 changes the operating point rather than the step time and is
probably the largest *aggregate throughput* lever of the four, because it
converts flat-frontier HBM directly into concurrent sequences.

A realistic target is **+10–12% from P2/P4**, with P1 and P3 as the two paths to
anything larger.

## Suggested experiment order

1. **No Booster allocation required.** (a) `benchmarks/kernels/` W4A16 MoE
   backend comparison at M ∈ {8,12,16,32} on the login-node GH200 — decides P1
   before any campaign; (b) extend the offline replay to an 8-sequence step to
   get the true concurrency slope for P3.
2. **P2** — draft-loop graph capture first (smaller, cleaner), then prologue
   batching. Gate on the golden 400K SHA
   `d594e4d4…dcfc528` plus the 24-prompt realistic suite.
3. **P3** — if the concurrency slope holds up, re-plan KV against a fixed
   ~3,000 hot slots/rank and qualify c=5 or c=6.
4. **P1** — only if step 1(a) shows a winning backend.
5. Re-profile and re-derive this budget before starting P4.

Offline replay proved cheaper than a cluster campaign for two of the three
original items. Any future placement or tiering hypothesis should be run through
`analysis/p1_balance.py` and `analysis/p2_tier_sweep.py` first; the span model
matches the measured trace to ~1%.
Analysis scripts used for this report are reproducible from the traces alone;
the segmentation rule (graph correlation IDs) and the tier rule (`add_` stream)
are the only two non-obvious pieces.
