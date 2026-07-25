# AutoRound W4G64

Goal: compare `c-bf/GLM-5.2-AutoRound-W4G64-MTP` with the current
compressed-tensors W4G128 baseline on the same GH200 tiered-MoE runtime.

Pinned checkpoint:

- revision: `e1ba8871b936706b02212a862bcb3a0f33e0391b`
- local path: `../models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887`
- vLLM integration: `4e563c699`
- config SHA-256:
  `5c4de205561e37a0c5257061f9913ca350308332bafdc8328b4476ea7f69b67e`
- index SHA-256:
  `50a9de1f3897ab363dca6dcb27db37ba5c2a53bbf622a88d88c92b1e3450f0e5`

The routed experts are symmetric GPTQ-packed W4G64. The existing Marlin
execution path is unchanged; the loader now recognizes `qweight/qzeros/scales`,
discards unused symmetric zero points, casts FP16 scales to BF16, and streams
one expert at a time into group-64 hot/cold destinations.

The model card says this checkpoint was calibrated on 512 coding-agent
samples at sequence length 2,048 for 200 iterations. That makes it relevant
to this workload, but it is not evidence of an accuracy improvement by itself;
quality still needs a matched evaluation against the W4G128 baseline.

Header-derived c1/TP4/EP4/MTP1 plan at 400K context and 7 GB HBM reserve:

- runtime expert: 20,054,024 bytes
- non-routed weights per rank: 13,252,290,560 bytes
- hot/cold slots per rank without a placement profile: 2,614 / 2,186
- planned Grace expert allocation: 43,838,096,464 bytes per rank

The first smoke deliberately uses a 10 GB reserve and no inherited placement
profile. After correctness, copy the current placement mapping under the new
checkpoint fingerprint, cap it to the measured HBM edge, and run the matched
realistic prompt benchmark.

`hybrid-p0.5-profile.json` is the existing c1 Python/PyTorch/ML/math placement
with only the checkpoint fingerprint changed. It is an A/B control, not a new
AutoRound routing trace; the planner will demote it from 2,870 hot slots/rank
to the smaller W4G64 HBM budget.

## Smoke result

Job `1045456` completed successfully on `jpbo-032-42` with TP4/EP4, c1,
MTP1, 400K context, a 10 GB HBM reserve, and no placement profile.

- checkpoint load and tier conversion: 683.09 seconds
- routed expert streaming on rank 0: 4,800 experts in 357.49 seconds
- loaded model memory: 61.17 GiB per rank
- observed free HBM after warmup: 10.32-10.33 GiB per rank
- available HBM KV cache: 23.04 GiB / 400,064 tokens
- short MTP acceptance: 100% (8/8 drafted tokens)
- semantic completion: `Paris`
- Python completion contained a valid recursive Fibonacci implementation

The post-request `EngineDeadError` in `smoke-server.out` is expected teardown:
the job killed the server only after both HTTP requests returned 200 and their
JSON responses were saved.

Artifacts:

- `smoke-server.out` / `smoke-server.err`
- `smoke-semantic.json`
- `smoke-python.json`

## Matched c1/MTP3 result

Job `1045471` completed successfully on `jpbo-016-28`. It used the same 24
Python/PyTorch/ML/math prompts, 256 output tokens, concurrency 1, 10 GB HBM
reserve, MTP3, and capture size 4 as the W4G128 baseline.

| Metric | AutoRound r1 | AutoRound r2 | AutoRound mean | W4G128 mean | Delta |
|---|---:|---:|---:|---:|---:|
| Output tok/s | 91.73 | 92.45 | 92.09 | 95.44 | -3.51% |
| Total tok/s | 103.52 | 104.32 | 103.92 | 107.69 | -3.51% |
| Mean TTFT (ms) | 274.48 | 278.87 | 276.67 | 278.71 | -0.73% |
| Mean TPOT (ms) | 9.87 | 9.77 | 9.82 | 9.43 | +4.13% |
| MTP acceptance | 62.47% | 63.34% | 62.91% | 61.78% | +1.82% |
| MTP acceptance length | 2.87 | 2.90 | 2.89 | 2.85 | +1.18% |

All 48 measured requests completed with no errors and exactly 256 output
tokens. Spot checks across all four domains were coherent and on-topic, but
this is not an accuracy evaluation.

The group-64 checkpoint is therefore systems-viable. It costs about 3.5%
throughput in this matched run, while MTP acceptance is slightly higher. A
quality evaluation is required before deciding whether that trade is useful.
If retained, retrace expert routing on this checkpoint before final tuning;
the current placement is deliberately the W4G128 A/B control mapping.

Cold-load time varied strongly with GPFS state:

- smoke job `1045456`: 683.09 seconds
- benchmark job `1045471`: 1,193.60 seconds

Artifacts:

- `benchmark-r1.json` / `benchmark-r2.json`
- `benchmark-server.out` / `benchmark-server.err`
