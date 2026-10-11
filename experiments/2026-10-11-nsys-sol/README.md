# Nsight Systems SOL / roofline of the shipped config, c=1..16 (2026-10-11)

Where decode time goes on the current stack, per kernel against its roofline,
to choose the next work. Tree: `worktrees/fmla-p1` (vLLM 80bd3cce5c: one-shot DCP,
FlashMLA Phase 1 schedule; register-split FlashMLA library 9c2356f2).

- c=1: prod launcher config (`claude-glm53-sliced-df2-dcp4.sh`: DFlash2 k=7, 8
  tokens/step, TP-sliced, 400K).
- c=2/4/8/16: the MTP3 launchers (`claude-glm53-sliced-mtp3-c<c>.sh`: 4c
  tokens/step, pools 0.8M/0.8M/1.6M/1.6M, reserve 3.6/3.6/3.6/4.0, captures to 4c).
- One fresh hold per c (2272141-2272145, `launch.sh 1 2 4 8 16`), `nsys_arm.sh`
  with graph kernels traced per node (`nsys_node.sh`). Two 2 s windows per c
  (`prof_load_n.py`): c requests at ~5K context (w1) and at ~50K (w2).
- Outputs `/e/fscratch/profound/$USER/sliced-prof/nsol-c<c>-<job>/nsys.{1,2}.*`.
  Analysis in `an/`: `bd-*` (step breakdown, `../2026-10-09-tp-sliced-experts/
  nsys_bd.py`, now tolerant of reports with no memset table), `kc-*`
  (`kchain_nsys.py`: per-kernel us/step and the median layer chain), `st-*`
  (`staging_cost.py`), `sol-w{1,2}.md` (`sol.py`), `bd-w{1,2}.md` (`bd_table.py`).

The step period here is the GPU decode-step period (verify launch to verify
launch). It is below the A/B "step ms" (c16 50K: 49.8 vs 58.4), which also
absorbs time behind other requests' prefill.

## Step breakdown (ms/step, mean of 4 ranks, ~50K window; `an/bd-w2.md`)

Each instant goes to the highest-priority category running, so side-stream
work that overlaps the MoE kernel is not counted.

| category | c=1 | c=2 | c=4 | c=8 | c=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **step period** | **18.83** | **20.11** | **25.92** | **36.69** | **49.83** |
| MoE expert GEMMs (sliced, + shared expert) | 6.53 | 7.68 | 12.01 | 20.11 | 28.78 |
| dense GEMMs (decode_gemm + cuBLAS) | 4.19 | 4.19 | 4.48 | 5.17 | 5.54 |
| DCP collectives | 1.31 | 1.30 | 1.64 | 2.16 | 3.47 |
| outside the verify graph (draft, lm_head, sampling, prep) | 1.78 | 1.99 | 2.27 | 2.62 | 3.31 |
| all-reduce + RMSNorm (incl. wait) | 1.20 | 1.16 | 1.40 | 1.82 | 2.66 |
| attention (FlashMLA) | 0.85 | 0.85 | 1.08 | 1.43 | 1.92 |
| DSA indexer | 0.88 | 0.87 | 0.95 | 1.05 | 0.96 |
| GPU idle | 0.58 | 0.58 | 0.59 | 0.62 | 0.88 |
| other verify-graph kernels | 0.49 | 0.45 | 0.46 | 0.49 | 0.84 |
| MoE route/act/finalize + router | 0.49 | 0.50 | 0.54 | 0.60 | 0.71 |
| KV write / skip-KV staging | 0.26 | 0.26 | 0.26 | 0.29 | 0.45 |
| attention (split-KV combine) | 0.28 | 0.28 | 0.24 | 0.32 | 0.32 |

The ~5K window is within 0.1-0.7 ms of these. The exceptions are the indexer
(0.3 ms lower at 5K) and the staging-driven items below.

## Per-kernel SOL (~50K window; `an/sol-w2.md`, 5K in `an/sol-w1.md`)

The us/call figures are the mean of 4 ranks, with the per-call minimum over
ranks in brackets. PDL dependents (MoE finalize, FlashMLA combine) are charged
only their tail past the producer's end. Rooflines:

| resource | roofline |
| --- | --- |
| HBM | 3.6 TB/s |
| C2C | 0.419 TB/s |
| NVLink inbound | ~0.43 TB/s |
| bf16 | ~630 TFLOP/s |

Byte and FLOP models are as in `../2026-10-09-tp-sliced-experts` (c16 SOL
table), scaled by T tokens per step. MoE floors come from the per-call
max(hot/HBM, cold/C2C) at the mean routing mix of that README:

| c | floor (µs) | basis |
| --- | ---: | --- |
| 1 | 61.3 | DFlash2 38 hot / 4 cold |
| 2 | ~61 | interpolated |
| 4 | 119 | |
| 8 | 221 | |
| 16 | 378 | |

Dense = 8.92 GB of weights per step (o_proj 50.3 MB, fused qkv_a 32.5, q_b + indexer
wq_b 16.8 per call, router, W_UV, W_UK) over HBM = 2.48 ms.

| kernel group | c=1 us/call (ms/step) | % SOL | gap ms | c=4 us/call (ms/step) | % SOL | gap ms | c=16 us/call (ms/step) | % SOL | gap ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MoE layer_kernel (75) | 87.0 (6.53) | 70 | **1.93** | 160.4 (12.03) | 74 | **3.11** | 386.5 (28.99) | 98 | 0.64 |
| dense GEMMs | (4.28) | 58 | **1.80** | (4.59) | 54 | **2.11** | (5.54) | 45 | **3.07** |
| AR + RMSNorm (151-157) | 7.7 [5.6] (1.17) | 9 | 1.06 | 9.0 [6.4] (1.41) | 15 | 1.19 | 16.9 [12.2] (2.66) | 32 | **1.80** |
| FlashMLA sparse (78) | 10.9 (0.85) | 12 | 0.75 | 13.8 (1.08) | 20 | 0.86 | 24.6 (1.92) | 44 | 1.07 |
| DCP gather_cat (78) | 7.6 [6.4] (0.60) | 13 | 0.52 | 9.4 [7.9] (0.74) | 22 | 0.58 | 18.0 [16.9] (1.40) | 45 | 0.77 |
| DCP lse_reduce_scatter (78) | 6.8 [6.3] (0.53) | 13 | 0.46 | 8.1 [7.5] (0.63) | 22 | 0.49 | 18.3 [15.0] (1.42) | 39 | 0.86 |
| indexer, side stream (189) | 5.7 (1.08) | | | 6.6 (1.25) | | | 10.1 (1.90) | | |
| MoE route_prep + top-k (150) | 4.7 (0.71) | | | 4.8 (0.72) | | | 5.3 (0.79) | | |
| FlashMLA combine tail (78) | 3.6 (0.28) | | | 3.1 (0.24) | | | 4.1 (0.32) | | |
| MoE finalize tail (75) | 2.0 (0.15) | | | 2.6 (0.19) | | | 4.4 (0.33) | | |
| KV write (135) | 1.8 (0.24) | | | 1.9 (0.25) | | | 2.1 (0.28) | | |
| skip-KV staging, side stream (76) | 7.5 (0.57) | | | 17.7 (1.35) | | | 88.6 (6.73) | | |

Notes:
- **Collectives, FlashMLA and the small kernels at c ≤ 4** sit at 9-22% of a
  bandwidth roofline because they are latency-bound: 2-11 µs kernels with
  ~0.4-2 MB each. The mean minus the minimum over ranks is wait or skew:
  - AR: 2.1 µs/call at c1, 4.7 at c16, so 0.3 / 0.7 ms per step;
  - the DCP one-shots: ~1 µs at c1, 1-3 µs at c16.
  These kernels add up to ~4.9 ms/step at c1 (AR, DCP, FlashMLA and combine,
  route/top-k, KV write, other) over ~700 launches. Shrinking them needs fewer
  launches (fusion or a persistent layer kernel), not faster kernels.
- **MoE at c16 is at its floor (98%) and that floor is C2C.** At c ≥ 4 the
  cold-expert stream bounds the kernel. The residency at c8/c16 is lower
  (13.5-13.6K hot per GPU vs 14.3K at c2/c4 and 14.8K at c1) because the 1.6M
  pool takes HBM. So the floor itself grows with c. At c1/c2 the kernel is
  HBM-side and runs at 60-70% of its floor: task #69's lever 1.
- **Dense GEMMs** are the only kernel group 1.8-3.1 ms over roofline at every
  concurrency. At c ≤ 4 that is decode_gemm (skinny, M ≤ 16). At c16 it is all
  cuBLAS nvjet: o_proj 22.4 µs vs 14.0 floor, qkv_a 15.4 vs 9.0, q_b 9.3 vs 4.7.
- **FlashMLA after Phase 1**: 24.6 µs/call at c16 (39.2 before Phase 1),
  44% of the compute roofline. At c1 the 10.9 µs is ~5 µs of fixed ramp and
  split work.
- **Outside the verify graph** (1.8-3.3 ms), from the c16 trace on device 0:

  | item | ms/step |
  | --- | ---: |
  | MTP drafter MoE (Marlin, untiered, 6 calls of 127 µs) | 0.76 |
  | lm_head (target + 3 drafter, ~140 µs each, ~93% SOL) | 0.58 |
  | sampler `_topk_topp` (1 call) | 0.23 |
  | drafter's own sparse FlashMLA (3 x 72 µs, 3x the target's per-call time; not on the Phase 1 schedule) | 0.22 |
  | NCCL ring all-gather (4 x 52 µs) | 0.21 |

  At c1 (DFlash2) the drafter's cutlass GEMMs take 0.41 ms and lm_head 0.29.

## Skip-KV staging grows with c (`staging_cost.py`)

The 57 skip layers keep their MLA KV on Grace. Per group, the anchor forks a
side-stream copy of the selected rows to HBM: `_gather_rows_kernel`, 8 CTAs
per layer, over C2C. The design point was U ≈ 615 rows per group at c1, 8
queries (gate 0a, `../2026-10-06-skip-layer-kv-grace`). The gather duration
here, mean of 4 ranks:

| window | c=1 | c=2 | c=4 | c=8 | c=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5K (µs) | 17 | 21 | 34 | 63 | 194-227 |
| 50K (µs) | 19 | 25 | 56 | 114 | 284-341 |

- The c16 p90 is 400-590 µs.
- U is not logged: `VLLM_SKIP_KV_STATS` needs a run. At ~90-100 GB/s (8-CTA
  latency-bound) the c16 50K gathers are ~25-30 MB per group.

What it costs on the critical path:
- **Late skip layers** (FlashMLA waiting on the staged event after all other
  work has ended): none at c ≤ 8. At c16 it is 0.01-0.02 ms/step at 5K and
  0.03-0.20 ms/step at 50K (0.6-3.3 late groups per step).
- **Contention with the MoE kernel.** At c ≥ 4 the gathers overlap the next
  layer_kernel: 0.4 / 1.3 / 5.0 ms of overlap per step at c4 / c8 / c16 50K. A
  regression of layer_kernel duration on that overlap, demeaned by layer
  position and by step, gives a slope of 0.5 / 0.2-0.4 / 0.16-0.22 µs per µs.
  Estimated cost:

  | c | 50K window | 5K window |
  | --- | --- | --- |
  | 4 | 0.21-0.24 ms/step | ≤ 0.04 |
  | 8 | 0.27-0.54 | ≤ 0.03 |
  | 16 | 0.83-1.01 | ~0.1 |

  This is consistent with both drawing on C2C, where the MoE kernel is bound
  from c4 up. It is an estimate from the trace, not yet a controlled
  measurement.

## Where to focus next (ranked by gap at the target concurrency)

The gaps are against rooflines, so they are upper bounds and are not
additive. Values are ms/step at ~50K.

| # | lever | c1 / c2 | c4 | c8 | c16 | notes |
|--:|---|--:|--:|--:|--:|---|
| 1 | **Dense GEMMs to the HBM floor** (decode_gemm at M ≤ 16; cuBLAS at M = 32-64) | 1.8 | 2.1 | 2.8 | 3.1 | Largest gap at c ≥ 8 and second at c1. At M = 64, o_proj / qkv_a / q_b are 50-63% SOL. Candidates: skinny-GEMM v2 extended to M = 64 (it ends at 32), or fusing q_b into qkv_a's epilogue. Same math. |
| 2 | **MoE kernel to the HBM floor at small M** (task #69) | 1.9 / 3.1 | 3.1 | 3.6 | 0.6 | At c ≤ 2 it is HBM-side and 60-70% of floor. At c ≥ 8 it is at the C2C floor, so only fewer cold bytes help (#4). |
| 3 | **Launch-bound tail of each layer** (AR, DCP one-shots, FlashMLA ramp, combine, route/top-k, KV write, glue) | 2.8 (AR + DCP + FlashMLA gaps; ~4.9 ms of small kernels in all) | 3.1 | 3.6 | 4.5 | Each is 4-18 µs, latency-bound and 9-45% SOL. Fusing AR + DCP gather + combine, or a persistent attention tail, attacks the launch count. The megakernel estimate is the upper end. Includes 0.3-0.7 ms of AR rank skew. |
| 4 | **C2C budget at c ≥ 4: staging vs cold experts** | 0 | 0.2 | 0.3-0.5 | 0.8-1.0 (+0.03-0.2 late) | Make the staging gathers cheaper on C2C or move them off the MoE window. Options: stage only rows not already staged for the previous step (rows stay hot across steps for a request), or start the copy earlier. First measure U (`VLLM_SKIP_KV_STATS`) at c8/c16 and run a controlled probe (layer_kernel bench with and without a concurrent U-row gather). |
| 5 | **Drafter path outside the graph** | DFlash2 GEMMs 0.4 | 0.4 | 0.6 | 1.0 | MTP drafter MoE (Marlin untiered, 0.76 ms at c16) and the drafter's sparse FlashMLA without the balanced schedule (3 x 72 µs vs 24 µs for the target). Both are the same math run faster. |
| 6 | FlashMLA sparse (inside #3) | 0.75 | 0.86 | 1.0 | 1.07 | After Phase 1 it is at 44% of the compute roofline at c16. Remaining work: consumer math (the Phase 0 trace). |
| 7 | sampler, NCCL all-gather, idle | ~0.8 | ~0.9 | ~0.9 | ~1.3 | `_topk_topp` 0.18-0.23 ms (8 CTAs) and ring all-gather 0.2 ms at c16; idle 0.6-0.9 ms. |

At **c=1**, the prod config, the order is MoE (1.9), dense (1.8) and then the
launch-bound tail (~2.8 over roofline). At **c=16** it is dense (3.1), the
tail (~4.5 together, AR alone 1.8) and C2C contention (0.8-1.0). The MoE kernel
itself is done there unless cold bytes shrink.

## Caveats

- nsys traces per graph node. Kernel durations are close to un-profiled, but
  per-launch costs inflate eager regions (outside the verify graph).
- The c2 MoE floor is interpolated. The c8/c16 floors use the earlier
  residency (14.7K hot); the current one is 13.5-13.6K, so the true floors are
  a little higher and the c16 % SOL is a little lower than shown.
- The staging contention is a regression on trace data. It needs the
  controlled probe in lever 4 before acting on it.
