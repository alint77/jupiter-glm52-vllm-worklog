# MTP6 depth check

## Result

MTP6 is not useful for the target exact-400K workload. It reduces decode from
108.17 to 84.47 tok/s versus the qualified MTP3 result, a 21.9% regression.
The extra draft passes do not increase accepted tokens per target step at long
context, so MTP7 and MTP8 were skipped.

The server uses the same TP4/EP4, tiered HBM/Grace placement, 10 GB HBM
reserve, FP8 MLA cache, and deterministic decoding as MTP3. MTP6 changes only
the speculative depth and the compiled/full-CUDA-graph verification size from
four to seven.

| Context | Depth | TTFT | TPOT | Decode | Acceptance | Tokens/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | MTP3 | 0.848 s | 9.669 ms | 103.42 tok/s | 78.51% | 3.36 |
| 4K | MTP6 | 10.741 s* | 9.324 ms | 107.26 tok/s | 55.83% | 4.35 |
| 399,744 | MTP3 | 114.999 s | 9.245 ms | 108.17 tok/s | 60.74% | 2.82 |
| 399,744 | MTP6 | 110.158 s | 11.838 ms | 84.47 tok/s | 31.33% | 2.78 |

`*` The MTP6 4K TTFT includes remaining first-request Triton JIT and is not a
qualified TTFT result. Its steady decode measurement is usable. The historical
4K random prompt was not identical, so its small 3.7% decode improvement is
directional rather than a matched-prompt comparison.

## Why it loses at 400K

For the exact same seed-13, 399,744-token request, the profiled MTP3 run drafted
261 tokens and accepted 167 across 87 target steps. MTP6 drafted 517 and
accepted only 162 across 91 steps. It therefore performs 98% more draft work,
accepts 3% fewer draft tokens, and lowers acceptance length from 2.92 to 2.78.

| Draft position | MTP3 acceptance | MTP6 acceptance |
| ---: | ---: | ---: |
| 1 | 94.25% | 81.32% |
| 2 | 62.07% | 48.35% |
| 3 | 35.63% | 25.27% |
| 4 | n/a | 10.99% |
| 5 | n/a | 7.69% |
| 6 | n/a | 4.40% |

The last three passes almost never survive verification, while each still
runs the FP8 MTP layer, full-context indexer path, synchronization, vocabulary
projection, and all-gather. This matches the earlier roofline: the MTP core is
small, but additional serial draft passes are not free at 400K.

MTP6 does not create a capacity problem. Its size-7 graph uses 1.90 GiB, leaves
about 9.2 GiB free HBM per GPU, and retains a 400,064-token KV capacity. The
failure is efficiency, not memory.

## Correctness

The MTP6 exact-400K output is byte-for-byte identical to the matched MTP3
profiled request. Both generated texts have SHA-256
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`.
This validates greedy rejection sampling at depth six; the result is slower,
not incorrect.

## Decision

Keep MTP3 as the fixed-depth default. MTP6 may help short contexts on some
prompts, but a global depth increase is wrong for the 400K objective. If depth
is revisited, it should be dynamic and acceptance/context aware rather than a
fixed MTP7 or MTP8 setting.

MTP7 had only begun startup when it was cancelled; no MTP7 or MTP8 performance
result is claimed. Job `968901` was released after the MTP6 decision.

## Reproduction

Launch [run-server.sh](run-server.sh) inside a four-GPU Booster allocation,
then run [run-sweep.sh](run-sweep.sh) on the same node. Raw benchmark results
are in [mtp6-4k-256.json](mtp6-4k-256.json) and
[mtp6-399744-256.json](mtp6-399744-256.json).
