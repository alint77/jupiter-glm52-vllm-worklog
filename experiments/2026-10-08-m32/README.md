# Everything gated on M <= 8, extended to M <= 32 (2026-10-08)

c=2 / c=4 with DFlash2 k=7 makes 16- / 32-token verify steps. Every
decode-specific path was built for 8:

| gate | was | for 9-32 tokens the server ran |
|---|---|---|
| tiered decode MoE (`tiered_decode`, `MAX_TOKENS`) | 8 | the wgmma prefill MoE kernel |
| `decode_gemm` (`MAX_M`) | 8 | cuBLAS |
| skip-KV stager (`MAX_QUERIES`) | 8 | the 57 skip layers' FlashMLA reading their KV from Grace over C2C |
| replica balancer (`tiered_replica_max_tokens`) | verify x max_num_seqs | already scales |

## Hot / cold experts per call at M = 8 / 16 / 32 (`m32_grid.py`)

Live Claude Code capture (76 files, 41,055 8-token steps); a 16- / 32-token
step = 2 / 4 concurrent requests (8-token steps from distinct files, random);
prod placement (frequency profile, 3,670 hot / GPU, its replicas, the deployed
balancer); 6,000 steps x 75 layers x 4 GPUs per M (`grid-m8-16-32.txt/json`).

| M | hot / GPU-layer p5 / p50 / p95 / max | cold p5 / p50 / p95 / max | cells covering 95% | experts with > 8 tokens |
|---|---|---|---|---|
| 8 | 5 / 9 / 15 / 26 | 0 / 1 / 2 / 8 | 38 | 0 |
| 16 | 11 / 16 / 24 / 41 | 0 / 1 / 4 / 10 | 73 | 0.26% |
| 32 | 19 / 26 / 37 / 52 | 0 / 3 / 6 / 13 | 130 | 1.15% (max 31) |

## Tiered decode MoE to 32 tokens

An expert still takes at most 8 tokens per list entry (tokens are the mma N
dimension; a 32-token stage would not fit shared memory next to 4 stages of
weights). `MAX_TOKENS` 8 -> 32 sizes the workspace (routes, token rows: ~6 MB);
`route_prep` counts each expert's routes first and opens ceil(n / 8)
consecutive entries, route `pos` going to entry `pos / 8`. The GEMM kernels
accumulate per route (w13) and per token (w2) with fp32 atomics, so a split
expert is exact; it re-streams the expert's weights per extra entry (1.15% of
experts at M=32). Tests (`tests/kernels/moe/test_tiered_decode_moe.py`, now
also T = 16 / 32, with 8-16 tokens per expert, and the replica assignment at 32):
14 passed; error vs fp32 3.6-3.9e-3 (Marlin 7.1-8.8e-3).

**Kernel bench** (`bench_moe_m32.py`, `grid_moe.sh`, node jpbo-015-32, one GPU
per M NUMA-bound, every 95%-cell, 20 routings per cell drawn with GLM's
tokens-per-expert distribution at that M, graph replay, 2 rounds alternating;
round-to-round spread 0.2-0.6%), us per layer weighted by cell frequency:

| M | decode kernel | wgmma prefill kernel (what ran) | per layer | x 75 MoE layers |
|---|---|---|---|---|
| 16 | 153.5 | 203.3 | -49.8 (-24.5%) | ~-3.7 ms/step |
| 32 | 251.4 | 297.7 | -46.3 (-15.6%) | ~-3.5 ms/step |

Faster on every cell (prefill / decode p10 1.15 at 16, 1.09 at 32); the
smallest gains are the cold-heavy cells (26 hot / 7 cold at M=32: 398 vs 422).

## Skip-KV stager to 32 queries

`MAX_QUERIES` 8 -> 32 as a cap; buffers sized by `limits()`: queries =
max_num_seqs x (1 + speculative tokens) (8 at prod c=1, 32 at c=4 k=7),
columns = the DCP decode index width (768) instead of 2048, since each staged
row is one distinct index entry. 32 x 768 x 656 B x 3 = 48 MB at c=4 (32 MB
at c=1 today, 8 x 768 now: 12 MB). `stage()` asserts the indices fit. Cached,
so the planner's call at load fixes what the stager allocates. Tests: staging
bit-identical at 1 / 8 / 32 queries x width 2048 / 768, graph replay; 49 passed.

## decode_gemm to 32 tokens (`bench_gemm_m32.py`, 2 runs on Booster)

1-4 token tiles per CTA, each weight fragment feeding all; 16-row CTAs (32- /
64-row CTAs to cut L2 re-reads of x never won). Best config vs cuBLAS:
o_proj -18% (8) / -17% (16) / -15% (24) / -9% (32); fused qkv_a -18 / -14 /
-9 / +2; q_b -5 / -2 / +6 / +12; dense gate_up -16 / -15 / -13 / -11; down
-13 / -12 / -9 / -5. `config(n, k, m)` keeps cuBLAS for q_b past 16 tokens
and fused qkv_a at 32. Tests M = 1-32: 41 passed.

## M = 8 unchanged (`grid_m8.sh`, 3 rounds alternating)

New decode kernel 91.93 us/layer vs HEAD 92.04 over the 38 cells; per cell
-0.8% .. +0.7%.

## Served A/B: c=4, DFlash2 k=7 (`chain_conc.sh`, `compare_conc.py`)

3 nodes x 4 arms alternating, before = 2af85c93c3 (worktree via PYTHONPATH),
after = 32a592fc1a; conc_probe --quad, 4 reps; step = acceptance / per-request
tok/s, after - before paired per node (+- standard error):

| case | tokens/step | before | after | diff | total tok/s before -> after |
|---|---|---|---|---|---|
| alone 5K / 50K | 8 | 22.40 / 22.41 | 22.43 / 22.40 | +0.03 / -0.01 | 150 / 136 -> 146 / 139 |
| pairs (3 mixes) | 16 | 32.92-33.12 | 29.62-29.79 | -3.30 to -3.33 (+-0.05-0.25) | 192-202 -> 207-222 |
| quads (3 mixes) | 32 | 46.44-47.46 | 42.10-42.76 | -4.34 to -4.81 (+-0.11-0.27) | 273-277 -> 299-304 |

## Concurrency sweep (`conc_sweep.py`, `sweep_arm.sh`, `launch_sweep.sh`)

New code, one node per config, 3 reps, n in flight = 1, 2, 4, 8 (<= c) at 5K
and 50K (averaged here), 400 tokens; per request / total decode tok/s:

| config | hot | 1 | 2 | 4 | 8 |
|---|--:|--:|--:|--:|--:|
| MTP3 c=2 (reserve 3.6) | 3550 | 165 / 153 | 127 / 228 | | |
| DFlash2 k=3 c=2 | 3591 | 158 / 148 | 125 / 223 | | |
| MTP3 c=4 (reserve 3.6) | 3382 | 154 / 143 | 125 / 221 | 96 / 331 | |
| DFlash2 k=3 c=4 | 3429 | 148 / 138 | 121 / 213 | 95 / 331 | |
| MTP3 c=8, 200K | 3381 | 155 / 144 | 123 / 218 | 94 / 322 | 65 / 439 |
| DFlash2 k=3 c=8, 200K | 3427 | 147 / 138 | 118 / 214 | 93 / 323 | 65 / 432 |
| MTP3 c=8, 400K, r500 | 3044 | 132 / 123 | 101 / 182 | 78 / 274 | 55 / 369 |
| DFlash2 k=3 c=8, 400K, r500 | 3108 | 131 / 123 | 101 / 182 | 78 / 281 | 54 / 367 |
| MTP3 c=8, 400K, no replicas | 3044 | 129 / 121 | 98 / 172 | 72 / 260 | 51 / 346 |
| DFlash2 k=3 c=8, 400K, no replicas | 3108 | 124 / 116 | 97 / 172 | 72 / 253 | 51 / 340 |

Failures on the way: MTP3 c=2 / c=4 at reserve 3.0 (2.63 GB free, 2.71
required; ran at 3.6); `max_num_seqs` was pinned to 1-4 (vllm 0f21793f47
allows 8); c=8 x 400K with the r2000 replicas exceeds paired Grace capacity
(the planner had already demoted to 3,180 hot; fewer hot experts would add
cold ones to Grace). r500 = `profiles/glm53-w4a16-agentic-3239-r500-ccfreq3676.json`
(the prod profile's replicas re-placed at a budget of 500 per GPU).

## c=8 on a shared 1.6M pool, and r1000 (2026-10-09)

vllm c31885ae2d `VLLM_TIERED_MOE_KV_POOL_SEQS`: the KV pool in max_model_len
sequences, independent of max_num_seqs (runtime allocation, pooled path and
planner all read `kv_pool_seqs`). c=8, 400K, pool 4 (1.6M tokens), prod
r2000 replicas; and the 3.2M pool with the r1000 profile
(`profiles/glm53-w4a16-agentic-3239-r1000-ccfreq3676.json`). Per request /
total tok/s, 5K and 50K averaged:

| config | hot | 1 | 2 | 4 | 8 |
|---|--:|--:|--:|--:|--:|
| MTP3, 1.6M pool | 3381 | 152 / 142 | 126 / 226 | 97 / 337 | 66 / 444 |
| DFlash2 k=3, 1.6M pool | 3428 | 144 / 134 | 120 / 216 | 96 / 333 | 65 / 430 |
| MTP3, 3.2M, r1000 | 3044 | 135 / 126 | 106 / 191 | 81 / 285 | 55 / 377 |
| DFlash2 k=3, 3.2M, r1000 | 3108 | 134 / 125 | 103 / 189 | 81 / 279 | 55 / 376 |

The 1.6M pool matches the 200K-context runs (same hot set and replicas) at
full 400K per request; the write-up drops the 200K and no-replica rows.
