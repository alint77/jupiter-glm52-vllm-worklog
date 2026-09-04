# Phase 1: is the Grace->HBM staging copy fast, and is it quiet?

**Verdict: the premise survives.** Job 1667272 on `jpbo-012-43`, GH200 120GB,
`agent_space/benchmarks/cold_prefetch_dma.py`.

## Results

| size | h2d_pinned | d2d_uva | sm_read | NUMA local |
| ---: | ---: | ---: | ---: | ---: |
| 670 MiB | 417 GB/s | 418 GB/s | 420 GB/s | 1.000 |
| 830 MiB | 417 GB/s | 418 GB/s | 421 GB/s | 1.000 |

| size | Marlin alone | Marlin + copy | copy under load |
| ---: | ---: | ---: | ---: |
| 670 MiB | 2.840 ms | 2.939 ms (**+3.5%**) | 196 GB/s (47% of solo) |
| 830 MiB | 2.840 ms | 2.940 ms (**+3.5%**) | 221 GB/s (53% of solo) |

## What this settles

**The transfer mechanism does not matter.** `cudaMemcpyAsync` from pinned CPU,
a device-to-device copy from the CUDA alias, and an SM-issued elementwise read
all land within 1% of each other at **417-421 GB/s**. C2C is the limit, not the
engine, so the plan's worry that the copy-engine path might differ from the
SM-load path the project's 373 GB/s was measured on is answered: it doesn't.
**418 GB/s is 93% of the 450 GB/s spec**, and better than the 373 figure.

**The copy loses about half its bandwidth under concurrent compute, and the
compute barely notices.** 196-221 GB/s against 418 solo, while Marlin slows
only 3.5%. The contention is asymmetric, which is the right direction for this
design: the thing being hidden absorbs the cost.

**It still hides, with 3.3x margin in the worst case.** Recomputed at the
*concurrent* rate, not the spec rate:

| layer | H2D needed | vs window min 12.96 ms | vs median 16.11 ms |
| --- | ---: | ---: | ---: |
| median, 670 MiB | 3.58 ms | **3.62x** | 4.49x |
| largest, 830 MiB | 3.94 ms | **3.29x** | 4.09x |

Per chunk that is 52.1 GB at ~196 GB/s = **266 ms of DMA**, a 17% duty cycle
against a post-prefetch ~1530 ms chunk. The 3.5% compute penalty applies only
while the copy runs, so the chunk-average penalty is nearer **+0.6%, about
+9 ms** -- against a 373 ms saving, **net -364 ms**.

**NUMA locality was not the risk here.** The local fraction is 1.000 both bound
and unbound on this node, so the `numactl` binding changed nothing. Keep the
binding -- it has mattered before -- but it is not what makes this work.

## The finding that most strengthens the case

A **contiguous** read of the same Grace alias reaches **420 GB/s**, while cold
Marlin's own demand reads of the same memory achieve **91 GB/s logical**. That
is a **4.6x gap on the identical link and the identical data**, so the cold
tier's cost is its *access pattern*, not C2C bandwidth. Staging converts a
91 GB/s scattered read into a 418 GB/s contiguous one, which is precisely the
mechanism the design is built on.

It also bears on the open question from the profile: cold at a 4x M-block
re-read would move 210 GB per chunk, and at 418 GB/s that is 503 ms against the
583 ms observed -- consistent, though this run does not separate re-reads from
poor coalescing, and it does not need to.

## Caveats

* The Marlin load here is 64 sequential dense `marlin_gemm` calls at m=256,
  k=6144, n=4096 -- shaped like one layer's routed work, but not the real MoE
  kernel with its routing and tier split. It is representative of the HBM and
  SM pressure, which is what the contention question needs.
* The concurrent copy rate was measured with Marlin running *continuously*. In
  the real design the copy overlaps a mix of attention, all-reduce and hot MoE,
  so ~196 GB/s is conservative.

## Next

Phase 2: slot allocation and the copy path behind the flag, cold tier still
reading Grace, so the plumbing is verified without changing results.
