# TP-sliced experts: every expert 4-way on its intermediate dim (2026-10-09)

Idea (`../2026-10-09-first-principles`): instead of expert parallelism (each
GPU owns whole experts; the slowest GPU's MoE sets the step), GPU r holds rows
[512 r, 512 r + 512) of every expert's gate / up and the matching columns of
down, for hot (HBM) and cold (its own Grace) experts alike. Every GPU computes
its slice of every touched expert for all tokens; the all-reduce that already
follows the MoE sums the partials. Work per GPU is the node mean by
construction, each cold expert is read over all four C2C links at once, and
replicas / balancer / cost table go away.

`tiered_decode.cu`: the production kernel with build-time `TD_INTER` (2048 or
512), per-projection K chunk / ring depth (`TD_CHUNK0/1`, `TD_STAGES0/1`) and a
`TD_NO_FLUSH` timing probe; defaults rebuild today's kernel. Kept here (the
shared tree serves running A/B arms). `bench_slice.py --check`: the four 512
slices of the same checkpoint experts, summed, against the 2048 kernel and
fp32; `--time`: whole experts at per-GPU counts vs slices at node-wide counts
(graph replay of 20 distinct routings, best of 10). `analyze.py`: per-variant
linear fits, replayed over agentic held-out steps (EP = slowest GPU after the
deployed replica balancer, prod profile at 3,670 hot).

## Round 1 (job 2247626)

Check: slices summed vs fp32 2.2e-3 (whole kernel 2.5-2.7e-3), M = 8 and 32.

| MoE ms / step (75 layers) | M=8 | M=32 |
|---|--:|--:|
| EP, whole experts (1024 chunk, 4 stages), slowest GPU | 8.72 | 22.37 |
| EP, mean GPU | 7.49 | 20.28 |
| TP-sliced, 512 chunk x 8 stages | **8.11** | 22.41 |
| TP-sliced, 512 chunk x 4 stages | 11.84 | 34.47 |

Slices pay ~20% more per byte than whole experts (M=8: 44 hot slices 105.9 us
= 9.6 us per expert-equivalent vs 12 whole hot 96.6 us = 8.0), but cold is far
cheaper (M=32: 112 hot + 12 cold slices 297 us; 28 hot + 8 cold whole experts
on one GPU 447 us). Suspects: the down projection (K = 512) flushes its fp32
atomics after every 16 KB unit (4x the atomics per GPU), and the 512 chunk
on gate / up. Round 2: per-projection chunks, a 6-stage down ring, and
`TD_NO_FLUSH` probes.

## Kernel dev, slice regime (2026-10-09)

Dev loop: `kdev.py` (build with ptxas report + SASS; roof; bench; once; check),
`ab.sh` (same-GPU comparison, 4 GPUs, alternating order) + `ab_sum.py`,
`ncu1.sh` + `stall_regions.py`, `tl2.py` (CTA trace), `sass_loops.py`,
`tma_probe.py`. Nodes: the `develbooster` reservation (30 min holds,
`keep_holds.sh`); the main queue was blocked by a large planned job.

**Roofline** (`kdev.py roof`, plain loads; `tma_probe.py`, TMA ring):
HBM 2.42 TB/s at 132 x 64 KB in flight, 3.43 TB/s at 264 blocks; the
kernel's own TMA box patterns stream at 3.5-3.8 TB/s with 3-4 stages
(w13 box 3.76, contiguous 3.91), so layout and boxes are not the limit. C2C
~11 GB/s per CTA with plain loads, 419 GB/s at >= 48 CTAs.

| version | change |
|---|---|
| v1 | one persistent kernel: w13 -> per-entry activation (last contributor) -> w2 |
| v2 | + shared expert (bf16 TP slice) as units: swizzled TMA, ldmatrix, bf16 mma |
| v3 | Marlin-style INT4 decode, phase-specialised fully unrolled consumer (58 -> 26 instr per group step) |
| v4 | dynamic per-tier work queues (S0, R0, S1, R1 groups), stealing across tiers |
| v5 | route_prep fused in (lists by CTA 0, rows by CTAs 1..T, shared w13 from raw rows): slower, dropped |
| v6 | v4 with the ring addressed off `smem` (v2-v5's uintptr alignment had turned every smem load into a generic LD) |
| v8 | v6 + per-stage flush records in smem (rows, folded scales) written by the producer: no global loads in flushes |

Same GPU, us per call (shared expert included; v0 = the shipped kernel with
INTER 512 + the shared expert as cuBLAS on a side stream, today's shape):

| M=8 cell (hot/cold slices) | 30/2 | 38/0 | 38/4 | 46/6 |
|---|--:|--:|--:|--:|
| v0 + side shared | 93.9 | 102.7 | 115.5 | 135.9 |
| v6 | 90.8 | 106.4 | 111.8 | 133.5 |
| **v8** | **81.5** | **96.5** | **103.6** | **126.3** |
| v8 roofline share | 0.61 | 0.64 | 0.59 | 0.60 |

| M=32 cell | 96/8 | 110/0 | 110/12 | 124/16 |
|---|--:|--:|--:|--:|
| v0 + side shared | 274.7 | 266.8 | 323.6 | 368.2 |
| **v8** | **251.4** | **262.9** | **292.5** | **338.6** |

Cold-preferring CTAs: 16 best (8 collapses with cold slices, 24 / 32 slightly
worse).
