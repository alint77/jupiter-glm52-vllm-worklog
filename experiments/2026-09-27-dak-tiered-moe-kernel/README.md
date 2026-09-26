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

## Next

- Consumer energy per byte is the limit (clock around 1.6–1.8 GHz under full load):
  - fewer ALU ops in the dequant (reg_b costs 6);
  - cheaper group scaling;
  - 128-bit shared loads.
- Pick cold CTAs from (hot, cold) on the device; the current value is a host argument.
- Run the probability-weighted grid against the Marlin fits.

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
