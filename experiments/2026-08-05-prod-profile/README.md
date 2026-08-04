# Production profile: prefill, decode, and the mixed step

Status: **Complete.** Four torch-profiler captures of the production
configuration on the 16K PyTorch coding shape from
[`2026-08-05-pytorch-16k-c1-c4`](../2026-08-05-pytorch-16k-c1-c4/README.md).

The result that should drive the next work is the **mixed step**: when a chunked
prefill shares an engine step with decode, the step leaves the CUDA graph
entirely — 160 of its 166 custom all-reduces fall back to the un-graphed
two-stage kernel — and 61% of it becomes communication. Since that workload is
71% prefill-bound by wall clock, this is the largest lever the trace exposes.

## Captures

Job `1240390`, node `jpbo-004-02`, traces under
`/e/project1/profound/alint77/traces/prod-profile-1240390`.

| Capture | Contents | Steps/rank |
| --- | --- | ---: |
| `prefill-c1` | two 8,192-token chunks of one 16K prompt | 2 |
| `decode-c1` | steady-state c1 decode | 27 |
| `decode-c4` | **mixed**: the tail of prefill plus c4 decode | 11 |
| `mixed-c4` | **pure c4 decode** | 102 |

The last two are named for what they were meant to hold, not what they hold.
`wait_for_decode` polled `prompt_tokens_total` for stability, but that counter
only advances when a prefill *chunk completes*, and a chunk takes ~2.6 s — so
two polls 0.25 s apart read as "stable" in the middle of one. `decode-c4`
therefore opened its window inside the end of prefill, and `mixed-c4`, gated
behind `max_num_seqs=4` with four requests queued, never saw an admission. Both
are still usable, and the accident is what produced the mixed-step measurement.
`analyze_mixed.py` splits `decode-c4` by step wall time at 200 ms to separate
the two regimes.

**Config note.** The first attempt (job `1240192`, kept as
`v1-runner-profile-1240192`) did not set `VLLM_USE_V2_MODEL_RUNNER=1`, which
production does, so it profiled the V1 runner. Its numbers are retained only as
a V1/V2 comparison; every number below is V2.

## Profiler distortion, measured rather than assumed

| Regime | Profiled step | Unprofiled step (benchmark) | Overhead |
| --- | ---: | ---: | ---: |
| Decode c1 | 29.31 ms | 28.11 ms | +4.3% |
| Decode c4 | 43.24 ms | 42.9 ms | +0.8% |

Graph-replayed steps are essentially undistorted, so the decode absolutes below
can be quoted. Eager regions are instrumented per operation and are inflated;
treat the mixed-step milliseconds as an upper bound and its **structure** as the
finding.

## Decode, c1 — 28.1 ms/step real

| Bucket | ms | share |
| --- | ---: | ---: |
| routed experts (W4 Marlin) | 7.301 | 24.1% |
| TP/EP communication | 5.792 | 19.1% |
| dense/shared compiled GEMMs | 5.720 | 18.9% |
| glue, elementwise, uncategorized | 4.736 | 15.6% |
| GPU-empty host/graph gaps | 3.274 | 10.8% |
| attention (FlashMLA + DSA + KV) | 2.435 | 8.0% |
| MoE routing, activation, sum | 1.090 | 3.6% |

Hot 6.809 ms, cold 3.979 ms, sum 10.788 ms against a span of 8.096 ms — the
tiers overlap to within **1.287 ms** of `max(hot, cold)`, confirming Phase 32 on
production traffic. EP skew is **2.454 ms/step**, down from the 3.615 ms Phase
26 measured before replica assignment shipped.

Graph structure: **4 replays per step** — the target at 4,106 kernels plus three
draft graphs of ~80 — with 2.632 ms of in-graph idle, 0.642 ms between graphs,
and only 0.310 ms of un-graphed kernels. This is a marked improvement on the V1
structure Phase 26 recorded (7 graphs, 2.0 ms between-graph idle, 1.918 ms
un-graphed); **the V2 runner has already closed most of the sampling/logits
boundary** that phase called the next lever.

Top kernels per step, rank 0:

| ms | calls | kernel |
| ---: | ---: | --- |
| 11.234 | 317.3 | `marlin_moe_wna16::Marlin` |
| 3.684 | 172.1 | `cross_device_reduce_1stage` |
| 1.757 | 80.9 | `triton_tem_fused_mm_t_0` |
| 1.736 | 197.0 | `ncclDevKernel_AllGather_RING_LL` |
| 1.618 | 164.9 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` |
| 1.572 | 269.6 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` |
| 1.224 | 84.0 | `flash_fwd_splitkv_mla_fp8_sparse` |
| 0.791 | 77.8 | `_assign_kernel` (replica assignment) |

The 197 all-gathers plus 84 reduce-scatters are **DCP4 collectives, ~2.4 ms/step
or 8.5% of the c1 step**. DCP exists to shard the KV cache so four 400K agents
fit; at concurrency one it buys nothing and costs that.

## Decode, c4 — 42.9 ms/step real

| Bucket | ms | share |
| --- | ---: | ---: |
| routed experts (W4 Marlin) | 21.224 | 44.5% |
| TP/EP communication | 7.550 | 15.8% |
| dense/shared compiled GEMMs | 6.034 | 12.7% |
| glue, elementwise, uncategorized | 5.454 | 11.4% |
| attention (FlashMLA + DSA + KV) | 3.231 | 6.8% |
| GPU-empty host/graph gaps | 3.001 | 6.3% |
| MoE routing, activation, sum | 1.162 | 2.4% |

Same kernel census as c1 — 166 one-stage all-reduces, 306 Marlin GEMMs, 190
all-gathers — so the step is fully graphed. Routed MoE nearly triples from c1
while dense, glue and attention barely move: **the batch-1 fixed cost is
amortized, and at c4 the MoE is the step**. EP skew is 3.095 ms/step.

## Prefill — 2.62 s per 8,192-token chunk

| Bucket | ms | share |
| --- | ---: | ---: |
| routed experts (W4 Marlin) | 888.9 | 34.0% |
| TP/EP communication | 587.5 | 22.4% |
| attention (FlashMLA + DSA + KV) | 533.8 | 20.4% |
| glue, elementwise, uncategorized | 286.6 | 10.9% |
| dense/shared compiled GEMMs | 150.3 | 5.7% |
| GPU-empty host/graph gaps | 116.5 | 4.5% |
| MoE routing, activation, sum | 54.7 | 2.1% |

Two chunks per 16K prompt, 2.05 s and 3.19 s — the second attends over the
first. Prefill is eager by construction: the token count exceeds
`overlap_max_tokens`, so `apply_tiered` runs the hot and cold tiers serially
rather than on two streams. Attention is 20% here against 8% in decode, and
`GPU-empty` is 4.5% of a 2.6-second step, which is 117 ms of host time per
chunk.

## The mixed step — where the workload actually loses

Splitting the `decode-c4` capture at 200 ms:

| | pure decode | prefill-bearing |
| --- | ---: | ---: |
| Rank-steps | 40 | 4 |
| Mean wall | 43.24 ms | **760.50 ms** |
| GPU busy | 44.65 ms | 556.45 ms |
| GPU empty | 3.00 ms | **184.99 ms** |
| TP/EP communication | 7.550 ms (15.8%) | **455.855 ms (61.5%)** |
| routed Marlin | 21.224 ms (44.5%) | 78.827 ms (10.6%) |
| attention | 3.231 ms | 7.330 ms |
| `cross_device_reduce_1stage` per step | **166** | **6** |
| `ncclDevKernel_AllGather` per step | 190 | 212 |
| `marlin_moe_wna16` per step | 306 | 306 |

The census line is the finding. A pure decode step issues 166 one-stage custom
all-reduces, all inside the CUDA graph. A prefill-bearing step issues **six**.
The other 160 become `cross_device_reduce_2stage`, which the graph-gap analysis
finds **outside every graph replay**, along with 16.4 ms of all-gather, 7.2 ms
of Marlin and 2.7 ms of reduce-scatter — 50.6 ms/step of un-graphed kernels and
15.0 ms/step where the host sits in no CUDA call at all.

So a mixed step is not "a decode step plus some prefill". It is a step that
**falls out of the graph, switches all-reduce implementation, and spends 61% of
itself in communication** — which at four ranks with thousands of prefill tokens
routing to unbalanced experts is mostly waiting.

## Ranked targets

For the 16K coding workload, weighted by the 71%/29% prefill/decode split of c4
wall clock:

1. **Keep prefill-bearing steps inside the CUDA graph, or stop mixing.** The
   all-reduce switching to a two-stage un-graphed kernel is the mechanism. Two
   directions worth pricing: force the one-stage path at mixed-batch sizes, or
   schedule prefill chunks into their own steps so decode steps stay graphed.
   This is also the fix the benchmark's 1.8 s stalls asked for.
2. **Prefill's 22% communication and 20% attention.** Prefill is a third of the
   node's time on this shape and has never been profiled before this capture.
   Nothing here is yet costed.
3. **Run c1 traffic on DCP1.** 2.4 ms/step, 8.5% of the c1 decode step, is DCP4
   collectives that buy capacity only concurrency 4 uses.
4. **Dense GEMMs plus glue: 10.5 ms/step at c1, 37% of the step.** Three nvjet
   and Triton GEMM variants and ~735 elementwise launches. Batch-1 fixed cost
   that does not amortize until c4.
5. **In-graph idle, 2.63 ms/step.** Phase 24 priced node fusion at ~1.2% and
   recommended against a campaign; unchanged.

Retired by this trace: the sampling/logits boundary Phase 26 flagged is largely
gone under V2 (0.642 ms of between-graph idle against 2.0 ms), and hot/cold
overlap remains within 1.3 ms of its floor.

## Files

| File | What |
| --- | --- |
| `job-profile.sh` | Server + capture job, production config plus profiler |
| `capture.py` | Drives the four windows, waits on engine state |
| `analyze_prod.py` | Decode step budget via launch-correlation attribution |
| `analyze_prefill.py` | Same budget without the decode-only tier statistics |
| `analyze_mixed.py` | Splits a capture into decode and prefill-bearing steps |
| `step-budget-decode.json` | c1 and c4 decode budgets |
| `step-budget-prefill-c1.json`, `step-budget-mixed-c4.json` | prefill, pure c4 decode |
| `step-split-c4.json` | The mixed-step split |
| `graph-gaps.json` | Graph replay structure and un-graphed kernels |
| `v1-*` | The V1-runner first attempt, retained for comparison |
