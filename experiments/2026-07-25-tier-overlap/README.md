# P1b Phase A — why the hot and cold MoE tiers do not overlap

Diagnosis phase of the
[P1b plan](../2026-07-25-c4-mtp3-critical-path/p1b-implementation-plan.md).
Job 1041013, Booster node, GH200 120GB, 132 SMs, Grace pages 100% NUMA-local.
Shape: M=16, 19 activated hot experts, 3 cold, `cold_share=0.13` — the
trace-derived c4 MTP3 operating point.

## A0 — the positive control that was missing last time

`blocks_per_sm` is honoured by `marlin_mm` only when `thread_k` and `thread_n`
are also supplied; otherwise `determine_exec_config` discards it. The previous
sweep passed `(bps, -1, -1)` throughout and therefore compared identical
launches. This run reads the launched grid straight out of a profiler trace:

| config | grid | expected `sms * bps` | |
| --- | ---: | ---: | --- |
| `bps=-1, tk=-1, tn=-1` (auto) | 396 | — | |
| `bps=1, tk=64, tn=128` | **132** | 132 | APPLIED |
| `bps=2, tk=64, tn=128` | **264** | 264 | APPLIED |
| `bps=3, tk=64, tn=128` | **396** | 396 | APPLIED |

Control **PASS**: three distinct grids. Everything below is measured with the
knob provably live.

## A1 — the auto thread tile is (64, 128)

| config | time | Δ auto |
| --- | ---: | ---: |
| auto | 112.76 us | — |
| `bps=3, tk=64, tn=128` | 113.12 us | **+0.32%** |
| `bps=3, tk=128, tn=64` | 120.96 us | +7.27% |
| `bps=3, tk=128, tn=128` | 132.78 us | +17.75% |
| `bps=1, tk=64, tn=128` | 156.42 us | +38.72% |

`thread_k=64, thread_n=128` reproduces auto to within 0.32%, consistent with the
c4 trace (grid 396, 128 threads, and 64x128/64 = 128). Pinned for A2–A4.

Independent timing control: at that tile, `bps=1` vs `bps=3` differ by 38.3%.

## A2 — SM partitioning does not work (this time for real)

Solo, with the tile pinned:

| tier | bps=1 | bps=2 | bps=3 |
| --- | ---: | ---: | ---: |
| hot | 141.2 us (49.9% HBM roof) | 117.2 us (60.1%) | **114.0 us (61.8%)** |
| cold | 105.7 us (87.5% C2C roof) | 108.9 us (84.9%) | **105.1 us (88.0%)** |

Overlapped, all nine combinations:

| hot bps | cold bps | union | over ideal | captured |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 3 | **194.2 us** | +70.3% | 23.7% |
| 3 | 2 | 194.4 | +70.5% | 26.2% |
| 3 | 1 | 197.8 | +73.5% | 20.7% |
| 2 | 2 | 203.2 | +73.5% | 21.0% |
| 1 | 1 | 247.9 | +75.6% | −1.1% |

**The existing 3/3 default is the best of the nine**, and every combination sits
at +70–76% over ideal. H1 as originally posed — "partition the SM budget" — is
refuted, now with the control to back it.

The reason is visible in the solo column: cold is insensitive to `bps`
(105.7 / 108.9 / 105.1) because it is C2C-bandwidth-bound and has only 3
populated blocks of work, while hot degrades badly at `bps=1` (114 → 141 us)
because it has ~19 blocks x 32 N-tiles to spread. Taking block slots away from
hot costs more than it buys.

## A3 — the tiers really are barely co-resident

CUDA-event intervals per stream:

| case | union | overlap | **co-resident** | over ideal |
| --- | ---: | ---: | ---: | ---: |
| auto | 274.1 us | 30.4 us | **21.4%** | +68.8% |
| pinned 3/3 | 269.5 | 29.4 | 20.9% | +70.5% |
| pinned 2/1 | 272.7 | 56.8 | 37.1% | +54.8% |
| pinned 1/1 | 311.0 | 55.8 | 36.6% | +45.2% |

Verdict: **H1-leaning — the kernels barely co-reside** (mean 28.8%). Note that
partitioning *does* raise co-residency (21% → 37%) and lower over-ideal
(+70% → +45%), so the occupancy story is real; it just cannot be exploited
through `blocks_per_sm`, because the same knob cripples hot.

## A4 — the fix: issue the hot tier first

| variant | union | **co-resident** | over ideal | vs default |
| --- | ---: | ---: | ---: | ---: |
| cold-first (production) | 272.6 us | **22.0%** | +69.7% | — |
| **hot-first** | **240.1 us** | **85.9%** | **+10.1%** | **−11.9%** |
| cold-first, low-priority cold | 273.0 | 21.4% | +69.5% | +0.2% |
| cold-first, high-priority cold | 273.4 | 27.3% | +71.7% | +0.3% |

Swapping the issue order takes co-residency from **22% to 86%** and the union to
within **10% of ideal**. Stream priority does nothing either way.

The mechanism is consistent with A2: both tiers launch a full occupancy wave, so
whichever is enqueued first claims the block slots. `apply_tiered` issues cold
first, and cold's grid is mostly *empty* blocks (3 populated of 396) parked
ahead of the hot tier, which then cannot get going. Leading with hot — which has
real work for all 396 blocks — lets cold's few populated blocks backfill as
hot's retire.

## Outcome

Phase A answers the plan's exit question. SM partitioning (Phase B/C) is dead;
do not plumb a per-tier launch config. The structural single-launch fallback
(Phase D) is not needed yet either.

The fix is a **one-line reorder** in `apply_tiered`
(`vllm/model_executor/layers/fused_moe/modular_kernel.py`): issue `run_tier(0)`
on the main stream before `run_tier(1)` on the aux stream. The fork/join and the
`cold_output.add_(hot_output)` semantics are unchanged, so output must stay
byte-identical.

Expected production effect: ~12% off the routed MoE union. Against the 25.7 ms
layer span that is ~3 ms/step, roughly 5% of the 62.43 ms step — well short of
the 12.5 ms upper bound the plan quoted, because hot's own kernel lengthens when
cold co-resides. Qualification is job 1041016 against the post-draft-sync
baseline (realistic c4 warmed 185.05 tok/s, 4K c4 99.13/110.25, 396K c4 median
ITL 58.13 ms), gated on the semantic smoke and the exact-400K golden SHA.

## Caveats

Absolute times differ between A2 (plain timing) and A3/A4 (CUDA events around
each stream, which add instrumentation and include queueing), so **do not
compare across stages** — only within one. The A4 −11.9% is an
event-instrumented comparison of two variants measured identically, which is
what makes it usable.

`over_ideal` is relative to `max(hot, cold)`, and hot's measured time itself
changes between variants, so it is a within-row diagnostic rather than a
cross-row ranking. Union is the honest metric.

---

# Phase A conclusion after qualification: P1b is refuted at its root

The A4 hot-first reorder was implemented and qualified (job 1041016). It
**works as intended and changes nothing**, because the problem it fixes does not
exist in production.

## The reorder took effect

Applying the lesson this plan wrote down — verify the change reached the
hardware — the trace confirms the issue order flipped:

| | draft-sync (cold-first) | tier-order (hot-first) |
| --- | ---: | ---: |
| `hot_start − cold_start`, median | +2.05 us | **−2.08 us** |
| hot starts first | 2.0% of layers | **98.7%** |
| routed layer span | 25.869 ms/step | 25.627 ms/step (**−0.94%**) |

So the change is live and its effect is ~1%, inside noise. End-to-end it is
flat: realistic c4 warmed −0.34%, 4K c4 r1 +8.2% against r2 −3.6% (variance,
not signal), 396K c4 median ITL −2.4%. The golden SHA reproduces.

## Why: the tiers were already overlapping

Measuring hot/cold co-residency directly in the production traces:

| | baseline (cold-first) | tier-order (hot-first) |
| --- | ---: | ---: |
| **co-resident fraction of layer union** | **81.5%** | 80.0% |
| hot chain, in situ | 329.8 us | 275.2 us |
| cold chain, in situ | 281.4 us | 327.2 us |
| layer span | 343.8 us | 341.7 us |
| serial would be | 611.2 us | 602.4 us |
| ideal `max(hot, cold)` in situ | 329.8 us | 327.2 us |

**Production overlap is already 81%, and the layer span is within 4% of the
in-situ ideal.** The isolated benchmark's "22% co-resident, +70% over ideal" was
an artifact of *eager* launch, where the first kernel's grid is submitted before
the second launch call returns. Under CUDA-graph replay both tiers are dispatched
by graph topology and already run concurrently. A4 fixed a benchmark artifact.

Note the roles simply swap under hot-first: hot drops 329.8 → 275.2 us and cold
rises 281.4 → 327.2 us, with the span unchanged. That is the signature of a
zero-sum contention effect — whichever tier goes first runs faster, and the
other absorbs the contention.

## The 12.5 ms headroom was an accounting error

That figure came from comparing the production layer span (25.7 ms) against
**isolated solo kernel costs** (13.2 ms). The implied ideal assumes each tier
runs at its solo speed *while overlapped*. Two kernels sharing SMs and memory
pipelines cannot both run at solo speed simultaneously; the in-situ hot chain is
329.8 us against ~176 us isolated precisely because cold is running alongside it.

The correct scheduling bound is the in-situ one: span 343.8 us against
`max(hot, cold)` = 329.8 us, i.e. **~4% headroom, not 48%**. Everything beyond
that requires making the kernels faster or reducing the work, not rescheduling
them.

This is the same class of error as the "39–40% overlap saving" corrected
earlier: comparing quantities measured under different contention conditions.
The rule that would have caught all three: **never compare an in-situ time with
an isolated time and call the difference recoverable.**

## Disposition

The reorder is **reverted** — it is neutral and adds diff without benefit.
P1b is closed as refuted. The routed MoE is not scheduling-limited; the
remaining levers are P3 (spend HBM on KV, buy concurrency) and P4 (graph node
count), plus anything that genuinely reduces routed work.
