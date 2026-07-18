# MTP verification overlap and local argmax

This experiment implements the first two priorities from the MTP profile:

1. run the hot-HBM and cold-Grace routed experts concurrently for verification
   sizes up to four tokens;
2. let the DeepSeek/GLM MTP draft head select its token locally, reducing the
   vocabulary collective to one value/index pair per rank.

The vLLM change is commit
[`5000658c4`](https://github.com/alint77/vllm/commit/5000658c4) on
`alint77/vllm:tiered-moe-grace-mtp`. Local argmax remains controlled by
`use_local_argmax_reduction`; it is not silently enabled.

## Exact-400K serving results

Each value below is one 399,744-input/256-output batch-one request with seed 13,
greedy decoding, ignored EOS, TP4/EP4, mode-3 compilation, and a CUDA graph for
the exact verification size. Up to eight Booster nodes ran the matrix in
parallel. Load-time GPFS contention does not enter these request timings.

| Configuration | Run 1 | Run 2 | Mean decode | Mean acceptance |
| --- | ---: | ---: | ---: | ---: |
| MTP3, serial tiers | 108.17 | - | 108.17 tok/s | 60.74% |
| MTP3, overlap | 125.96 | 129.39 | 127.67 tok/s | 60.57% |
| MTP3, overlap + local argmax | 128.53 | 124.73 | 126.63 tok/s | 61.81% |
| MTP2, serial tiers | 114.20 | 109.11 | 111.65 tok/s | 75.74% |
| MTP2, serial + local argmax | 111.51 | 110.55 | 111.03 tok/s | 74.51% |
| MTP2, overlap | 116.19 | 103.57 | 109.88 tok/s | 74.17% |
| MTP2, overlap + local argmax | 112.35 | 116.34 | 114.34 tok/s | 74.52% |

MTP3 overlap raises the two-run mean by 18.0% over the historical serial MTP3
control. MTP2 remains slower than optimized MTP3. Local argmax has no
repeatable batch-one throughput gain: the MTP3 pair is 0.8% slower on average,
while the MTP2 measurements are noisy and configuration-dependent.

The 4K random-prompt results are intentionally not used for the decision. Their
continuations and acceptance varied enough to move TPOT materially even with a
fixed random input seed.

## Trace evidence

Values are rank means over eight target steps. Cumulative routed time grows
under concurrent HBM/UVA contention, so the effective span is the union of all
routed-kernel intervals.

| Configuration | Routed sum | Overlap | Effective span | Draft gather grid/count |
| --- | ---: | ---: | ---: | ---: |
| MTP3 serial | 8.476 ms | 0.000 ms | 8.476 ms | 16, 24 calls |
| MTP3 overlap | 11.508 ms | 4.363 ms | 7.145 ms | 16, 24 calls |
| MTP3 overlap + argmax | 11.573 ms | 4.358 ms | 7.215 ms | 1, 24 calls |
| MTP2 serial | 7.413 ms | 0.000 ms | 7.413 ms | 16, 16 calls |
| MTP2 overlap | 10.270 ms | 3.778 ms | 6.492 ms | 16, 16 calls |
| MTP2 overlap + argmax | 9.976 ms | 3.630 ms | 6.346 ms | 1, 16 calls |

Overlap therefore removes 1.331 ms, or 15.7%, from the MTP3 routed span and
0.921 ms, or 12.4%, from the MTP2 routed span. The traces show three routed
stream IDs instead of one and concurrent hot/cold intervals in every layer.

For local argmax, each draft all-gather changes from a size-16 grid carrying a
38,720-element BF16 vocabulary shard to a size-1 grid carrying the local best
value/index pair. On TP4, the ring traffic falls from about 232 KB to 24 bytes
per draft position. Collective latency is already small, which explains why
the structural reduction does not produce a stable batch-one speedup.

Profile directories:

- MTP3 serial: `/e/scratch/profound/naeimitabiei1/glm52-mtp-profile-968182`
- MTP3 overlap: `/e/scratch/profound/naeimitabiei1/glm52-mtp3-overlap-profile-969112`
- MTP3 fast path: `/e/scratch/profound/naeimitabiei1/glm52-mtp3-fastpath-profile-969113`
- MTP2 serial: `/e/scratch/profound/naeimitabiei1/glm52-mtp2-profile-969109`
- MTP2 overlap: `/e/scratch/profound/naeimitabiei1/glm52-mtp2-overlap-profile-969389`
- MTP2 fast path: `/e/scratch/profound/naeimitabiei1/glm52-mtp2-fastpath-profile-969390`

## Correctness and decision

All configurations completed CUDA graph capture and the 400K request. Every
exact-400K continuation has SHA-256
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`.
An independent semantic prompt also produced the identical eight tokens
` Paris. Distance from Paris to Lyon is` in all eight parallel runs.

The focused local-argmax test and all 32 tiered manifest tests pass. Ruff,
formatting, typos, mypy, project source guards, and the signed-commit check pass.
The full MTP test file has two passing local tests; its other two cases require
downloading `XiaomiMiMo/MiMo-7B-Base`, which is unavailable in the offline test
environment.

Retain MTP3 as the default depth and retain verification overlap for all sizes
up to four. Keep local argmax as an opt-in implementation for larger batches or
future collective-heavy configurations; do not enable it by default from these
batch-one results.
