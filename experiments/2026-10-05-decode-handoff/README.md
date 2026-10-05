# Decode MoE: where the ~15 us per call goes, and whether 10 us of it is reclaimable (2026-10-05)

Question (user): the cold-only roofline
(`../2026-10-04-c2c-roofline/README.md`) leaves ~10-16 us per call above the
421 GB/s link floor; if at least 10 us of it can be reclaimed, implement it.

**Decision: not implemented.** The measured timeline puts the reclaimable
part at ~5-6 us per call cold-only and ~7-8 us in mixed cold-bound calls.
The one cheap knob found (w2's pre-wait weight prefetch) moves whole-call time
by < 1 us. Details and the remaining option below.

## Method

`bdf4eebec9` extends the `TD_CTA_TRACE` probe build: route_prep / act /
finalize blocks {entry, past PDL wait, exit}, the gemm producers' {first
issue, PDL return}, consumer warp 0's {first, last full stage}. Production
SASS is identical to HEAD (cubin diff), kernel tests 9/9.
`../2026-10-04-c2c-roofline/gap_probe.py` replays 20 calls back to back in one
CUDA graph (fresh experts per call; 1000 hot / 100 cold pool), 99 calls per
cell, medians. Hold 2184449.

## Timeline (us from the call's route_prep start)

Cold-only c = 2 (call 117.3, link floor 100.9, gap 16.4) and mixed (4, 2)
(call 122.0, gap 21.1). Link-idle stretch in bold.

| stretch | c = 2 | (4, 2) | what |
|---|--:|--:|---|
| route_prep -> first w13 cold issue | **3.65** | **4.1** | lists needed before any cold TMA; route_prep 2.6-3.1 + w13 ready/prologue ~1 |
| w13 stream vs its floor | +3.5 | +4.2 | first issue -> first full stage 1.4-3.4; steady state at ~420 GB/s |
| w13 last data -> w13 end | **1.7** | **1.7** | last stage math + red.add flush |
| w13 end -> w2 first issue | **2.0** | **2.2** | w2 launch on freed SMs + prologue |
| w2 stream vs its floor | +2.2 | +4.7 | ramp; in mixed calls rows arrive late (below) |
| w2 last data -> next call | **3.5** | **4.25** | drain 1.7, finalize ready +0.7, finalize 0.6-1.3, launch 0.5 |

Raw: `../2026-10-04-c2c-roofline/gap-run1.jsonl`, `gap-run2.jsonl`; per-CTA traces (70 MB) in `/e/fscratch/profound/naeimitabiei1/traces/2026-10-05-decode-handoff/`.

## The w13 -> w2 handoff (act)

From w13's end, us (`handoff.py`):

| | act end | w2 PDL return (rows can load) |
|---|--:|--:|
| cold-only (0, 2) | 2.5 | 2.9 |
| mixed (4, 2) / (9, 1) / (9, 2) / (9, 0) | 5.2-6.4 | 7.5-8.3 |
| mixed, no w2 weights issued before the PDL wait (hot and cold) | 3.0-3.3 | 3.4-3.7 |
| mixed, hot issues none, cold keeps 4 stages | 3.0-3.4 | 3.6-7.1 |

act blocks run ~2x slower, and the PDL return lags act's last exit by 2-4 us,
while w2's pre-wait weight prefetch is in flight (hot: up to 108 CTAs x 4 x
34 KB = 15 MB HBM; cold: 24 x 4 x 34 KB over C2C). Without the prefetch the
handoff matches cold-only, but w2 then starts its stream cold. Whole-call A/B
(normal builds, best of 2 interleaved rounds, `ab.sh`, `ab-hotpre.jsonl`):
hot pre-wait stages 4 / 2 / 1 / 0 are within +-3 us of each other in every
cell, which is the bench's run-to-run spread (cold-only cells, which the knob
does not touch, move by up to 4 us). In the trace build the no-prefetch
variant is -0.7 us at (4, 2), -0.7 at (9, 0), -0.5 at (9, 1), +0.1 at (9, 2).
Reverted.

## What is reclaimable

Only link-idle time that a different structure removes:

- **w13 drain + handoff + w2 ramp** (~5-6 us cold-only, ~7-8 mixed): cold
  CTAs run straight from their w13 units into w2 weight streaming (w2's
  weights need only route_prep's lists), with the act rows gated per expert
  by a flag. The 4-stage ring (~7.8 us of link at 24 CTAs) covers drain + act
  + flag latency. This is v1 of `../2026-10-04-tiered-decode-pair-pipeline`
  restricted to what the trace shows is idle.
- mbarrier init / tensormap prefetch before w13's PDL wait: ~0.5 us.
- finalize folded into w2's last CTA: ~1 us.
- Not reclaimable inside a call: route_prep's 2.6-3.1 us (no cold address
  before the lists exist), the ~1.5 us first-access latency of each stream,
  w2's final drain.

Total ~6.5-9.5 us per cold-bound call, < 10. At ~2 cold per GPU per layer
that is ~0.5-0.7 ms of a 24.5 ms step (2-3%), only on layers where the cold
tier is the binder, for a kernel restructure with new cross-CTA flags (the
pair-pipeline plan's hazards: reset discipline, ring head-of-line, graph
replay). Left for the user to decide.

## Files

- `ab.sh`, `ab-hotpre.jsonl`: whole-call A/B harness and the prefetch-depth run.
- `handoff.py`: handoff medians from gap_probe traces.
- traces per variant in `/e/fscratch/profound/naeimitabiei1/traces/2026-10-05-decode-handoff/` (TD_HOT_PRE / TD_COLD_PRE were
  temporary defines, reverted).
