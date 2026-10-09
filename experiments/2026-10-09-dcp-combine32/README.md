# DCP attention combine one-shot at 32-token steps (2026-10-09)

vllm c1aec2e22f. `OneShotCollectives.lse_reduce_scatter` gated on (out + lse)
x world size < the custom all-reduce buffer (8 MiB), but eager stages only this
rank's out + lse and the captured path stages nothing. A 32-token GLM-5.3 verify
step ([32, 64, 512] bf16 + LSE = 2.1 MB, x 4 = 8.4 MB) was refused; the prefill
route is off under capture, so NCCL reduce-scatter + Triton correction ran
(`../2026-10-09-c8-profile`: 78 x 16.4 us + 78 x 2.9 us per step). The gate is
now the bytes actually staged (= the C++ check).

**Tests** (`check.sh`, compute node): `test_custom_all_reduce.py -k one_shot`
4 passed; the captured collectives test now includes a 32 x 64 x 512 combine
checked exactly against NCCL (refused before the fix at 4 GPUs).

**Profile** (c=8 DFlash2 k=3, 1.6M pool, 8x5K, rank 0;
`c8-profile/dflash2-k3-combine32`): no NCCL kernel or `_correct_attn_cp_out`
left; `lse_reduce_scatter_kernel` 78 x 11.6 us = 0.90 ms in their place (was
1.51 ms). Other verify-graph kernels 1.77 -> 0.67 ms, DCP collectives 2.25 ->
2.84.

**Same-node A/B** (`chain.sh`: 2 holds x 4 alternating arms, c=8 DFlash2 k=3,
1.6M pool, `../2026-10-08-m32/sweep_arm.sh`; before = c31885ae2d worktree;
`compare.py`, mean over pairs of per-arm median step ms):

| ctx | n | before | after | delta |
|---|--:|--:|--:|--:|
| 5K | 1 / 2 / 4 | 19.05 / 23.18 / 30.02 | 19.09 / 23.18 / 30.17 | +0.04 / 0.00 / +0.15 |
| 5K | 8 | 42.55 | 41.62 | **-0.93 (-2.2%)** |
| 50K | 1 / 2 / 4 | 19.40 / 23.68 / 31.71 | 19.44 / 23.97 / 31.48 | +0.04 / +0.29 / -0.23 |
| 50K | 8 | 44.17 | 43.39 | **-0.78 (-1.8%)** |

Only the 32-token step changes; 1-16 tokens within noise (the path there is
unchanged). Per-arm medians at n=8: 7 of 8 after-arms below their node's
before-arms. Total tok/s at 8: 449 -> 456 (5K), 426 -> 434 (50K).
