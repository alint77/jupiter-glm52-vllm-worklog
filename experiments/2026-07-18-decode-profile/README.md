# Phase 8 decode critical path

This phase profiles the selected Phase 7 configuration at exact 400K context,
then removes an unintended communication path with one model-local change.

## Profile

The PyTorch profiler skipped 49 chunked-prefill iterations and captured eight
decode steps of a 399,744-input/256-output request on all four ranks. Profiling
adds overhead, so its 31.04 ms TPOT is not used as a performance result; the
unprofiled Phase 7 reference remains 23.765 ms TPOT.

Per-rank GPU activity in milliseconds per decode token was:

| Rank | Routed Marlin | MoE support | NCCL | Custom AR | Dense GEMM | Sparse MLA | DSA indexer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.841 | 3.023 | 7.278 | 0.066 | 5.615 | 1.890 | 0.912 |
| 1 | 8.903 | 3.237 | 6.508 | 0.050 | 5.625 | 1.862 | 0.915 |
| 2 | 8.661 | 3.211 | 6.716 | 0.056 | 5.563 | 1.887 | 0.910 |
| 3 | 8.344 | 3.097 | 6.920 | 0.044 | 5.600 | 1.846 | 0.916 |

These categories overlap across streams and are not additive wall time. Rank 0
records 7 custom all-reduces, 150 NCCL reduce-scatters, 151 NCCL all-gathers,
and 300 routed Marlin calls per token. One all-gather is vocabulary output; the
other 150 reduce-scatter/all-gather pairs come from the 75 routed layers.

## Diagnosis and change

The Phase 7 collective estimate assumed all 157 hidden-state reductions used
the measured 4.06 us custom backend. In practice, vLLM automatically enabled
DeepSeek sequence-parallel MoE under TP4/EP4. Only the embedding and three
dense layers used the custom reduction; routed layers instead padded the
single decode token to four and used NCCL reduce-scatter/all-gather.

The v2 design keeps the batch-one hidden state mirrored across TP ranks. Each
rank executes only its owned experts through the existing expert map, then one
late TP all-reduce combines the partial outputs. Tiered DeepSeek layers now
explicitly select that existing non-sequence-parallel path. Non-tiered models
retain the previous automatic behavior. A temporary runtime probe confirmed
the replacement 1-by-6,144 BF16 tensors are contiguous, CUDA-graph capturable,
and accepted by custom all-reduce; the probe is not part of the implementation.

Raw four-rank traces, profiler tables, the machine-readable summary, launch
logs, benchmark JSON, and reproduction scripts are stored beside this report.

## Validation and result

The final server selected `MoEPrepareAndFinalizeNoDPEPModular`, compiled the
new graph in 133.61 seconds, and captured it in three seconds. The deterministic
smoke retained the exact eight output tokens:
` Paris. Distance from Paris to Lyon is`.

All performance runs use the Phase 7 reserve-10 placement, full/piecewise
graphs, deterministic random input, batch one, and 256 output tokens.

| Input / seed | TTFT | TPOT | Decode |
| --- | ---: | ---: | ---: |
| 4,096 / 13 | 0.836 s | 19.327 ms | 51.74 tok/s |
| 399,744 / 13 | 108.273 s | 18.097 ms | 55.26 tok/s |
| 399,744 / 14 | 109.024 s | 18.161 ms | 55.06 tok/s |
| 399,744 mean | 108.648 s | 18.129 ms | 55.16 tok/s |

The two exact-400K TPOT samples differ by 0.35%. Against the selected Phase 7
mean, TPOT falls 23.72%, decode rate rises 31.09%, and TTFT rises 1.53%. Against
the native CPU-offload baseline, TPOT falls 31.89% and decode rate rises 46.82%.
Post-request free HBM is 5,409-5,416 MiB per GPU, above the observed-memory
gate. This is the strongest result so far but remains below the 100 tok/s
project minimum.

The 31 dedicated tiered-loader tests, Ruff, bytecode compilation, JSON
validation, shell syntax checks, deterministic smoke, and both full-context
runs pass. A broader engine-argument run completed 115 tests and failed 14
unrelated cases because their fake model IDs are not cached while Booster runs
with Hugging Face offline mode.
