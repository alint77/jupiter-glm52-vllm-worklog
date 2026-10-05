# C2C read roofline vs the tiered decode kernel's cold tier

GPU 0 reading NUMA-local pinned Grace memory (`GraceAllocation`, the cold
tier's allocator), 2 GiB per measurement, best of 5. `c2c_roofline.py`, two
Booster runs (hold 2175795 -> `run.log`, `probe.sbatch` job 2176802 ->
`run-2176802.log`); the two agree within 1-3%.

## Ceiling: 421 GB/s

| CTAs | SM loads (16 B, 512 thr) | TMA bulk 32 KB x 4 (1 thread/CTA) |
| ---: | ---: | ---: |
| 8 | 180-188 | 420-421 |
| 16 | 310-320 | 421 |
| 24 | 388-392 | 421 |
| 32-48 | 414-419 | 421 |
| 132-264 | 402-407 | 417-420 |

Copy engine H2D: 421 GB/s. TMA reaches the ceiling from 8 CTAs with any
chunk/stage choice (16 KB x 8, 32 KB x 6, 64 KB x 3 all 421 at 24 CTAs), so
the decode kernel's 16 / 24 cold CTAs are not a link-bandwidth limit. The
390 GB/s we have been using as "the link" is ~7% under the real ceiling.

## The decode kernel, cold experts only

`tiered_decode_moe` with 0 hot + c cold experts (fresh experts every call,
CUDA-graph replay; call = route_prep, w13, act, w2, finalize):

| cold | us/call | floor at 421 GB/s | effective GB/s | of ceiling | gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65.1 | 50.4 | 326 | 77% | 15 us |
| 2 | 115.8-116.3 | 100.9 | 365-367 | 87% | 15 us |
| 3 | 162.9-167.4 | 151.3 | 380-391 | 90-93% | 12-16 us |
| 4 | 212.1-218.1 | 201.7 | 390-400 | 92-95% | 10-16 us |

The gap is a roughly constant 10-16 us per call, not a bandwidth shortfall:
the streaming itself runs at the ceiling, and the fixed part (route_prep,
w13 -> act -> w2 handoffs, pipeline fill of two weight streams, finalize) is
what's left. At the live traffic's 2.0 cold per GPU per layer that is ~15 us
x 78 layers ~ 1.2 ms of a 24.5 ms step if the cold tier is the critical path.
Not measured here: how much of that fixed part already hides under the hot
tier in mixed (h, c) calls.

## Mixed hot + cold calls (job 2184265, `mixed.sbatch`, `mixed-2184265.jsonl`)

Same bench, whole call, fresh experts every call from a 1000-expert hot pool
(HBM) and a 100-expert cold pool (Grace). "Alone" is the larger of the
hot-only and cold-only times; the link floor is c x 21.23 MB at 421 GB/s.

| hot, cold | us | alone (hot / cold) | over alone | link floor | of floor |
|---|--:|--:|--:|--:|--:|
| 4,1 | 69.4 | 65.2 (45.4 / 65.2) | +4.2 | 50.4 | 73% |
| 9,1 | 81.8 | 77.5 (77.5 / 65.2) | +4.3 | hot-bound | |
| 4,2 | 120.9 | 117.4 | +3.5 | 100.9 | 83% |
| 9,2 | 117.8 | 117.4 | +0.4 | 100.9 | 86% |
| 14,2 | 126.6 | 117.4 (108.6 / 117.4) | +9.2 | 100.9 | 80% |
| 4,3 / 9,3 | 171.2 | 163.4 | +7.8 | 151.3 | 88% |
| 14,3 | 172.2 | 163.4 | +8.8 | 151.3 | 88% |
| 14,1 | 117.1 | 108.6 hot | +8.5 | hot-bound | |
| 20,1 | 159.0 | 146.6 hot | +12.4 | hot-bound | |
| 20,2 | 172.3 | 146.6 hot | +25.7 | hot-bound | |

Hot-only: 45.4 / 77.5 / 108.6 / 146.6 us at 4 / 9 / 14 / 20 (~6.3 us per
expert, ~20 us intercept). Cold-only: ~51 us per expert (the link), ~14 us
intercept.

- Cold-bound mixes stay within 0-9 us of the cold-only call, i.e. 83-88% of
  the link floor at c = 2-3; the gap is the same fixed ~15 us. The
  pair-pipeline plan's "(9,2) w13 alone 127.5 us" does not reproduce: the
  whole (9,2) call is 117.8 us.
- Adding hot work costs most where it makes the call hot-bound: the cold
  tier holds 24 SMs (16 at c = 1), so the hot tier has 108 instead of 132,
  and (20,2) is +26 us over hot-only (~132/108 of the hot slope). TMA alone
  saturates the link from 8 CTAs, but the cold CTAs also decode and multiply
  (~26 GB/s per CTA at the hot tier's rate), so ~16 is their floor.
