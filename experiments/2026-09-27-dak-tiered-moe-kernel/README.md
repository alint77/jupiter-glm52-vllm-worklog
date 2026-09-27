# One-kernel tiered MoE GEMM for MiMo decode (DAK-style, TMA, GH200)

Goal: a single persistent kernel that streams hot experts from HBM and cold
experts from pinned Grace memory (UVA over C2C) together, instead of
production's two Marlin launches. The kernel is specialized for MiMo-V2 only:
- w13 is 4096x6144 and w2 is 6144x2048, both mxfp4 (e2m1 + e8m0 per 32).
- Decode sends at most 8 tokens per expert.
- The (hot, cold) counts come from the trace data.

All numbers below come from Booster GH200 nodes, one GPU, with host memory
NUMA-bound to that GPU's Grace node (`run_bound.sh`). None come from the login
node.

## Target workload (`count-distribution.txt`)

The distribution comes from 1500 held-out steps with the r1500 replica
placement, giving 414,000 (layer, GPU) cases.

- **Hot experts per GPU per layer:** median 9, p10–p90 5–14.
- **Cold experts per GPU per layer:** 0: 8%, 1: 26%, 2: 35%, 3: 21%, 4+: 10%.
- **Tokens per active expert:** 1 token in 74%, 2 in 17%, 3 in 5%.
- **Most likely cells:** (7–11 hot, 1–3 cold).

## What limits it

1. **The TMA side is solved.** `tma_probe.cu` shows:
   - One CTA with 128 KB of 1D bulk copies in flight pulls about 104 GB/s from Grace.
   - 16 such CTAs reach 421 GB/s of C2C while 116 CTAs still pull 3.54 TB/s from HBM (`probe-standby.out`).
   - The kernel's producer warp alone (DIAG=1) reaches 97–101% of HBM SOL.
2. **FP8 mma.sync is not native on sm_90.** ptxas lowers it to
   `F2FP.F16.E4M3.UNPACK_B` plus f16 HMMA (`tiered_moe_fp8.cu`, v4). The
   v4 hi/lo activation split doubled the HMMAs, so v4 was slower than bf16 v3.
3. **v5 (`tiered_moe_f16.cu`) keeps the unpack but drops the waste.**
   - It packs e2m1 as exact e4m3 bit placements, converts with one F2FP per register, and runs a native f16 mma.
   - The block scale is applied in fp32 once per 32-K group.
   - ncu then showed 19.6M shared-memory bank conflicts: the 8 token rows of x sat 2048 B apart, all on the same banks. A 32 B row pad fixed it, taking compute-only (DIAG=2) w13 at 33 hot from 157 to 125 µs (97% SOL).
4. **The GPU is power-capped.** Booster's GPU limit is 680 W (default 900).
   - An in-kernel clock64 vs globaltimer measurement puts the SM clock at 1942 MHz memory-only, 1765 MHz compute-only, and **1597 MHz full** (v5, w13 hot 33).
   - Load and compute therefore do not overlap for free: the full kernel took 159 µs while each half alone took 121–125 µs. Pipeline depth made no difference (3, 4 and 8 stages were identical).
   - Ablations show that **HMMA is the power hog**. Without it compute-only runs unthrottled at 1994 MHz. Without the dequant (HMMA kept) the clock falls to 1397 MHz.
5. **v6 (`tiered_moe_sk.cu`, current) changes four things:**
   - One-token experts run an f16x2-FMA GEMV instead of an mma that would waste 7 of its 8 columns. The f16 sum spans one 32-K group, then an fp32 scale-FMA is applied.
   - Stream-K: each tier's (tile, chunk) units are split into equal contiguous ranges per CTA. This removes the wave-quantization tail (9 experts × 64 tiles over 132 CTAs is 4.4 waves).
   - Each consumer warp flushes its partial sums with fp32 `red.global.add`, so there is no named barrier and no cross-warp reduction.
   - w13 now returns raw gate/up sums. silu×up belongs in w2's activation prep, which has to run anyway to build w2's f16 fragment layout.

   The shapes are compile-time MiMo constants, and each expert's token count is loaded one unit ahead.

## Results, v6 (µs per call, fp32 atomics output; token counts drawn from the trace mix)

| hot, cold (cold CTAs) | w13 | % SOL | w2 | % SOL | w13 + w2 | production Marlin fit |
|---|---|---|---|---|---|---|
| 9, 0 | 47.6 | 70% | 26.8 | 62% | 74 | ~112 |
| 9, 1 (16) | 53.2 | 62% | 28.9 | 57% | 82 | ~112 |
| 9, 2 (24) | 69.6 | 94% | 37.6 | 87% | 107 | ~115 |
| 9, 3 (24) | 101.4 | 97% | 53.6 | 91% | 155 | ~164 |
| 12, 2 (20) | 74.9 | 87% | 39.9 | 82% | 115 | ~144 |
| 6, 2 (24) | 69.7 | 94% | 37.6 | 87% | 107 | ~115 |
| 0, 2 (16–24) | 69.2 | 95% | 37.0 | 88% | 106 | ~115 |

The production Marlin column uses the decode-profile fit for w13+w2 with hot
and cold co-running: max(16 + 10.7·hot, 17 + 49·cold) µs
(`../2026-09-26-mimo-decode-profile`). That fit excludes the separate
act_and_mul and moe_sum kernels, which v6 folds in.

- **Cold-bound cells** (cold ≥ 2): C2C bounds both kernels, so v6 gains 7–20%.
- **Hot-bound cells:** v6 gains about 30%.
- **Cold CTA count:** 16–24 cold CTAs are needed to saturate C2C; 12 leaves it at ~72%.

Correctness is checked by `sk check`. It compares every route row (w13) and
every token sum (w2) against a dequantized fp32 reference, on hot-only,
cold-only and mixed MiMo cases, with both the GEMV and mma paths exercised.
Max relative error is about 3e-4.

## Fixed-clock regime (supersedes the power-cap tuning above)

The clock under sustained MoE load settles at about 1410 MHz: ~567 W on the
GPU, with the 680 W module cap (Grace included) and the SW power cap active.
We can't lock clocks without root, but `ncu --clock-control base` pins the SMs
at 1.41–1.52 GHz, the same regime. From here on, every comparison uses that
pinned clock, and kernels are optimized for it as the worst case:
- `fixedclock_sweep.sh` sweeps both kernels.
- `marlin_mimo.py` runs production `fused_marlin_moe` with MXFP4, MiMo shapes, the tight smem policy, hot SMs×2 / cold SMs×1 grids and the decode token mix.
- `ncu_sum.py` and `compare_fixedclock.py` summarize the results.

**Production Marlin per tier** (w13 + act + w2 + sum; align kernels, ~8.5 µs, excluded):

| active experts | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 16 |
|---|---|---|---|---|---|---|---|---|
| hot (µs) | 50.1 | 60.4 | 81.3 | 103.7 | 125.8 | 144.0 | 175.0 | 209.3 |

Cold: 1 → 79.9, 2 → 127.5, 3 → 179.3, 4 → 225.4 µs.

**Tiered v7 vs Marlin**, with Marlin taken as max(hot, cold), i.e. perfect overlap, its best case (`fixedclock-v7.json`):
- 1.5× in hot-only cells.
- 1.3–1.4× with 1 cold.
- 1.07–1.09× with 2–3 cold, where C2C bounds both kernels.

Trace-weighted over 81% of cases, Marlin takes 150.3 µs per layer and tiered 128.0 µs (1.17×), about 1.5 ms per decode step. The tiered time still leaves out its two activation-prep kernels.

**Current v7** (e5m2/PRMT decode, unconditional activation loads;
`fixedclock-v71.json`):
- Trace-weighted: Marlin 150.3 µs vs tiered 124.0 µs per layer (1.21×), about 1.8 ms per decode step.
- 1.4–1.6× when hot dominates, e.g. (9,1) 125.8 → 85.4 µs and (11,0) 144.0 → 90.4 µs.
- 1.06–1.09× in C2C-bound cells, e.g. (9,2) 127.5 → 116.9 µs.

## Accuracy (`acc_check.py`, a real MiMo expert, outlier-heavy activations)

| vs fp64 reference | rms rel | bf16 output equal to bf16(ref) |
|---|---|---|
| Marlin numerics (fp32 accumulate, bf16 out) | 1.67e-3 | 99.99% |
| v7, fp32 out | 1.1e-7 | – |
| v7, bf16 out | 1.67e-3 | 99.99% |
| v6 one-token f16x2-FMA path, bf16 out (removed) | 1.71e-3 | 86.5% |

The f16 FMA path was not lossless. At the pinned clock it was also no faster
than the mma (w13 at 9 hot: 52.4 vs 51.0 µs), so v7 uses the exact mma path for
every token count. The f16 cross-group accumulation experiment (1% gain) was
dropped for the same reason.

## Negative result: mixing tiers inside every CTA

Tried giving every CTA an equal, evenly interleaved share of both tiers, with a
separate accumulator per tier, so all 132 SMs share the dequant and the C2C
demand. At the pinned clock it was worse: w13 (9,2) took 82.3 µs vs 75.9
dedicated, and w2 48.6 vs 42.7.
- With every CTA pulling cold chunks, about 4.5 MB is queued on C2C, so each cold chunk waits ~11 µs. That is more than the 4-stage ring's ~7 µs of prefetch, and the in-order ring stalls behind it.
- The scheduler also cost 7% on hot-only runs.

Dedicated cold CTAs stay.

## What limits v7 at the pinned clock (w13, 9 hot / 33 hot)

| run | 9 hot | 33 hot |
|---|---|---|
| loads only (DIAG=1) | 36.9 µs | 129.6 µs (94% SOL) |
| compute only (DIAG=2) | 44.0 µs | 146.1 µs |
| full | 51.2 µs | 182.7 µs |

The consumer is the bottleneck, and overlapping it with the loads costs a
further ~25%.
- **ncu at the pinned clock:** issue active 55%, ALU pipe 61%, tensor 16%.
- **Stall samples:** ~58% are issuing, not-selected or math-throttle. The long-scoreboard stalls sit at the full-barrier TRYWAIT.
- **Issue costs** (`pipe_rate.cu`, cycles per warp-instruction per SMSP): F2FP 2.63, LOP3 2.06, PRMT 2.06, HMUL2 1.13, IMAD.HI 4.0.

Tried, and the reason each did or didn't help:

| change | result | why |
|---|---|---|
| e2m1 at e5m2 bit positions (value × 2^-14), f16x2 built by PRMT | exact; equal speed | kept as the default |
| unconditional x loads (stale rows only feed unflushed columns) | −3–4% hot | kept: drops 2 CS2R and a predicate per group |
| 16 consumer warps | compute-only 146 → 164 µs | a shared per-SM resource saturates |
| FP8 wgmma | not pursued | Hopper's FP8 tensor-core accumulator keeps ~14 bits (DeepSeek-V3), and activations would need a hi/lo split: lossy like the removed f16 path |

## Negative result so far: one fused layer kernel (`tiered_moe_layer.cu`, v8)

A single cooperative launch runs w13, then silu×up, then w2. Correctness is
verified end to end against fp32, including a second pass that confirms the
in-kernel reset; error is 2.3e-4 max relative.
- **Handoff:** w13 tiles interleave 32 gate rows with their 32 up rows. The warp that completes a tile queues it for a helper warp, which writes f16 activations with a per-(route, 32-group) scale, applied in fp32 next to the e8m0 weight scale. w2's producer streams weight chunks ahead and adds rows once the 32 tiles behind that K half are done.
- **Speed:** still slower than the two v7 kernels at the pinned clock:

| cell | fused | separate v7 |
|---|---|---|
| (9,0) | 121–142 µs | 78 µs |
| (0,2) | 127 µs | 115 µs |

- **Why:**
  - Per-flush fences and tile counters cost more than the one kernel boundary they remove.
  - act latency is 2–7 µs per tile.
  - The ring holds only ~8 µs of C2C prefetch across the w13→w2 transition.
  - Expert-major ordering, which should have made early experts ready early, made hot worse by splitting tiles across more CTAs.
- **Status:** kept for reference, not the path forward as is.

## Next

- **Cold-bound cells (60% of cases).** w13+w2 at 2 cold takes 115 µs against a
  98 µs C2C SOL. The fix is one fused layer kernel:
  - w13 is followed by the act-prep of each expert's route rows, run by the
    CTA that finishes that expert's last w13 unit;
  - then w2. The producers stream w2 weights, which don't depend on
    activations, while w13 drains, so C2C never idles between the GEMMs;
  - this also removes a launch and the prep kernels.
- **Hot cells.** At 1.45 GHz the consumer is compute-bound (w13 at 9 hot is 51 µs, 65% of SOL).

## Files

- `tma_probe.cu`, `probe.sbatch`, `probe-standby.out`: TMA bandwidth probe (HBM, Grace, co-run).
- The kernel versions:
  - `tiered_moe_v1.cu`, `tiered_moe_v2.cu`, `tiered_moe.cu` (v3): bf16 dequant with mma.
  - `tiered_moe_fp8.cu` (v4): FP8 mma.sync.
  - `tiered_moe_f16.cu` (v5): e4m3 unpack plus f16 mma, with ablation and profiling macros.
  - `tiered_moe_sk.cu` (v6): stream-K with the GEMV path.
- Build: `nvcc -O3 -arch=sm_90a -std=c++17 -lineinfo [-DPROF=1] [-DDIAG=1|2] -o sk tiered_moe_sk.cu`.
  - `DIAG=1` runs loads only; `DIAG=2` runs compute only.
  - `PROF=1` prints the SM clock and the consumer/producer wait breakdown.
- `hold.sbatch`, `onnode.sh`, `run_bound.sh`, `bench_sweep.sh`: standby allocation and NUMA-bound runs.
  - Usage: `onnode.sh "run_bound.sh 1 <bin> bench w13 9 2 24 4 0"`.
- `ncu_f16_hot33.ncu-rep` (bank conflicts) and `ncu_f16b_hot33.ncu-rep` (fixed): ncu reports.
