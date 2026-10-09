# c=8 on the 1.6M pool: decode step and memory, MTP3 vs DFlash2 k=3 (2026-10-09)

`prof_arm.sh <tag> <spec>`: prod serve.sh at c=8 (`MAX_NUM_SEQS=8`,
`VLLM_TIERED_MOE_KV_POOL_SEQS=4`, SPEC_K=3, MTP3 at reserve 3.6), torch-profiler
windows from `prof_load.py` (1x5K, 4x5K, 8x5K, 8x50K; forced 400-token
outputs), allocator snapshots (`MEM_SNAPSHOT_DIR` probe), nvidia-smi per
second. Traces / snapshots: `/e/fscratch/profound/naeimitabiei1/c8-profile/<tag>/`.
`breakdown2.py` (rank 0): each instant of a step goes to the highest-priority
category running; `kernels.py`: per-kernel time in the verify graph.

## Step (ms, rank 0, mean over the window)

| | MTP3 1x | DF2 1x | MTP3 4x | DF2 4x | MTP3 8x5K | DF2 8x5K | MTP3 8x50K | DF2 8x50K |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| tokens verified | 4 | 4 | 16 | 16 | 32 | 32 | 32 | 32 |
| period | 19.86 | 19.27 | 31.74 | 33.75* | 44.09 | 42.04 | 44.80 | 42.92 |
| MoE expert GEMMs | 6.15 | 5.94 | 15.40 | 15.29 | 22.91 | 21.89 | 22.71 | 21.68 |
| AR + RMSNorm (incl. wait) | 2.55 | 2.49 | 3.05 | 3.03 | 3.71 | 3.69 | 3.99 | 3.94 |
| dense GEMMs (decode_gemm + cuBLAS) | 4.03 | 4.04 | 4.40 | 4.41 | 5.21 | 5.20 | 5.30 | 5.29 |
| outside verify graph (draft, sampling) | 2.11 | 1.72 | 2.24 | 1.92 | 2.63 | 1.86 | 2.68 | 1.85 |
| DCP collectives | 1.20 | 1.21 | 1.82 | 1.83 | 2.23 | 2.25 | 2.24 | 2.26 |
| attention (FlashMLA + combine) | 1.22 | 1.21 | 1.64 | 1.64 | 2.20 | 2.20 | 2.22 | 2.22 |
| other verify-graph kernels | 0.49 | 0.61 | 0.60 | 0.65 | 1.75 | 1.77 | 1.77 | 1.78 |
| KV write / skip-KV staging | 0.25 | 0.25 | 0.26 | 0.27 | 0.74 | 0.74 | 0.77 | 0.77 |
| DSA indexer | 0.56 | 0.55 | 0.59 | 0.59 | 0.63 | 0.63 | 1.03 | 1.02 |
| MoE route/act/finalize + router | 0.56 | 0.56 | 0.66 | 0.67 | 0.69 | 0.70 | 0.70 | 0.70 |
| GPU idle | 0.74 | 0.70 | 1.07 | 3.45* | 1.38 | 1.10 | 1.42 | 1.41 |

\* four 28-60 ms gaps (arrivals); per-step idle median 0.50 (DF2) / 0.57
(MTP3), period median 30.70 / 31.44.

- **MoE is half the step at 32 tokens and sits on the HBM floor for hot
  experts.** Fit over the kernel-bench cells (`../2026-10-08-m32` grid,
  weighted): us/layer = 14.6 + 6.29 x hot + 23.46 x cold at M=32 (8: 22.8 +
  5.67 h + 20.58 c). A hot expert (20.3 MiB) streams at ~3.4 TB/s; a cold one
  costs 3.7x as much over C2C. At 32 tokens a GPU-layer touches ~26 hot + 3-4
  cold: cold is ~25-30% of MoE time. Served 306 us/layer vs bench 251 (3,381
  hot here vs 3,670 in the bench placement).
- **DCP combine falls back to NCCL at 32 tokens: ~1.5 ms/step.** The new
  `ncclDevKernel_ReduceScatter` (78/step x 16.4 us = 1.28 ms) and
  `_correct_attn_cp_out_kernel` (0.23 ms) appear only at 32 tokens. The one-shot
  `lse_reduce_scatter` needs nbytes x world < the custom all-reduce buffer
  (`CustomAllreduce` max_size 8 MiB): [32, 64, 512] bf16 + LSE = 2.1 MB x 4 =
  8.4 MB. The prefill-buffer route is off during capture, so NCCL. A 16 MiB
  buffer for the DCP group (+8 MiB/GPU) would keep it one-shot.
- **Dense GEMMs +1.2 ms at 32 tokens**: q_b and fused qkv_a go back to cuBLAS
  by design (`decode_gemm.config`, level or better there in the bench);
  splitK nvjet at 10.6 us for qkv_a is near its stream floor.
- **Skip-KV gather** 46 us x 19 at 32 tokens (0.88 ms) vs 16 us at 8: scales
  with queries x width.
- **Drafters**: MTP3's three sequential passes cost 0.8 ms/step more than
  DFlash2's one at c=8 (2.63 vs 1.86); everything inside the verify graph is
  the same. Per token at 8x5K: MTP3 44.09 / (8 x 2.82) = 1.95 ms, DF2 42.04 /
  (8 x 2.78) = 1.89 ms; at 8x50K MTP3 accepts 3.05 vs 2.78 and wins (sweep).

## Memory (rank 0, GiB of 95.0)

| | MTP3 | DFlash2 k=3 |
|---|--:|--:|
| hot experts / GPU | 3,381 | 3,428 |
| expert storage | 66.95 | 67.88 |
| KV (HBM) | 6.22 | 6.17 |
| drafter-specific | MTP layer MoE as Marlin 1.13 | fp8 drafter 0.60 + 2 rotary caches 0.38 |
| cold prefetch slots (prefill only) | 2 x 567 MiB = 1.15 | 1.11 |
| workspace | 0.82 | 0.47 |
| live at startup / after serving | 87.90 / 87.90 | 88.27 / 88.27 |
| allocator reserved-free, startup -> after 8x50K | 0.60 -> 2.95 | 0.51 -> 2.86 |
| device free at the end | 0.72 | 0.47 |
| nvidia-smi peak (MiB) | 96,673 | 96,968 |

Nothing grows live during serving; the allocator caches ~2.35 GiB of prefill
transients (8 x 50K chunks) and the device ends 0.5-0.7 GiB free, as planned.
DFlash2's 47 extra hot experts are MTP's 1.13 GiB Marlin MTP-layer MoE minus
its own drafter. Levers here are small: the prefill-only cold slots (1.1 GiB,
~56 experts) and the drafter's rotary caches (0.38 GiB).
