# Compiled host-UVA cache tracer bullet

Four-rank tracer bullet on JUPITER allocation `958529` with the 400K main MLA
cache in paired Grace memory. Tiered MoE used 4,330 HBM experts and 470
pinned-UVA experts per rank.

## Compile diagnosis

The default compiled startup failed during warmup. Disabling CUDA graphs alone
did not change the failure. With `CUDA_LAUNCH_BLOCKING=1`, the first invalid
access was isolated to FlashInfer's TRT-LLM MNNVL fused all-reduce plus RMSNorm
operator, not tiered Marlin or the host cache.

Disabling only that compiler pass produced a working compiled server:

```text
--compilation-config '{"cudagraph_mode":"NONE","pass_config":{"fuse_allreduce_rms":false}}'
```

The model compiled for the dynamic `(1, 8192)` range in 82.26 seconds. Engine
profile, cache creation, and warmup took 114.29 seconds. The final cache had
6,250 blocks, exactly 400,000 tokens, and 19.06 GiB of host-UVA storage across
78 tensors per rank. All sampled host pages were local. The post-warmup audit
reported 4.60 GiB physical HBM free against the 3.73 GiB minimum.

## Results

A deterministic 5-input/8-output request completed in 2.652 seconds and then
1.655 seconds warm, producing identical text both times. The exact random
4,096-input/256-output benchmark completed without errors:

| TTFT | TPOT | P99 ITL | Decode throughput |
| ---: | ---: | ---: | ---: |
| 1.740 s | 248.75 ms | 277.24 ms | 3.93 tok/s |

This is far below the 37 tok/s native offload baseline. A separate Grace-only
Marlin probe traversed a 6,098,780,160-byte gate/up working set with a median
70.06 microseconds per expert, so cold expert reads do not explain the roughly
249 ms/token end-to-end decode time. This run established the regression but
did not isolate it. The subsequent HBM-cache A/B produced essentially the same
246 ms/token result, showing that launch-heavy no-graph execution, not cache
placement, dominated this short-context measurement.

The v2 host-cache gate requires no large HBM gather and p95 at most 0.5
ms/token/rank. Host-UVA remains ineligible for automatic placement because it
has not passed that standalone full-footprint gate. The next A/B run kept the
same compile and expert path but moved the main MLA cache to HBM.

After the 4K request, the CUDA allocator retained prefill buffers and physical
free HBM fell to about 1.84 GiB per GPU. Startup reserve reconciliation is
therefore not yet a worst-request high-water guarantee.
