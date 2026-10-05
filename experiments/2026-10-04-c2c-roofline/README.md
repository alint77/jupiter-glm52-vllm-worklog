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
