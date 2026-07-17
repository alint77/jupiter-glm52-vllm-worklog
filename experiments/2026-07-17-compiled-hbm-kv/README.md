# Compiled HBM-cache tracer bullet

Four-rank A/B on JUPITER allocation `958529` with the exact 400K main MLA
cache moved from paired Grace memory to HBM. The plan retained 3,279 hot
experts and placed 1,521 experts in pinned Grace memory per rank.

## HBM cache integration fix

The first startup reached cache planning and failed because the tier helper
used physical placement to distinguish main and indexer layouts. Both layouts
correctly share HBM in this scenario, so all 99 specs were misclassified as
indexer specs. The implementation now classifies semantic layout before
choosing physical placement. A regression test creates all 6,250 blocks and
asserts 99 HBM tensors totaling 21,576,000,000 bytes. The focused suite passes
28/28; Ruff and Python 3.12 mypy pass.

The corrected server compiled with CUDA graphs disabled and the known-bad
FlashInfer all-reduce/RMSNorm fusion disabled:

```text
--compilation-config '{"cudagraph_mode":"NONE","pass_config":{"fuse_allreduce_rms":false}}'
```

It created the exact 400K cache and passed the post-warmup physical audit with
4.50 GiB HBM free per GPU. The deterministic short output exactly matched the
host-cache result.

## Cache-tier A/B

| Main cache | TTFT | TPOT | P99 ITL | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Grace host-UVA | 1.740 s | 248.75 ms | 277.24 ms | 3.93 tok/s |
| HBM | 1.008 s | 245.60 ms | 275.09 ms | 4.02 tok/s |

Moving the full cache to HBM improves TTFT but changes steady decode by only
1.3%. The host sparse-cache path is therefore not the dominant source of this
no-graph result.

## Tiered Marlin isolation

The production W4A16 fused-MoE kernel was measured with incompressible weights,
exact native shapes, preallocated workspaces, full-footprint expert cycling,
and 100% local pinned-Grace pages:

| One routed layer | Median GPU time |
| --- | ---: |
| Native 64-expert HBM | 262 us |
| Native 64-expert Grace UVA | 267 us |
| Split HBM/HBM | 533 us |
| Split HBM/Grace | 523 us |

Split and native outputs match. Amortized wall time closely tracks CUDA event
time, so neither UVA reads nor Python dispatch accounts for the roughly 245
ms/token server result. The split path has a real fixed double-call cost, but
it predicts tens rather than hundreds of milliseconds across 75 routed layers.

The remaining gap was the launch-heavy no-CUDA-graph execution mode. Earlier
eager tracer bullets were similarly around 200 ms/token, while the 37 tok/s
native baseline used full/piecewise CUDA graphs.

## CUDA-graph result

The optimized run kept Inductor mode 3, enabled `FULL_AND_PIECEWISE` CUDA
graphs for capture sizes one and two, and disabled only the failing
all-reduce/RMSNorm fusion. Both piecewise graphs and the full decode graph
captured successfully. Estimated graph memory was 0.14 GiB per rank.

| Configuration | TTFT | TPOT | P99 ITL | Decode rate |
| --- | ---: | ---: | ---: | ---: |
| HBM cache, no graphs | 1.008 s | 245.60 ms | 275.09 ms | 4.07 tok/s |
| HBM cache, full/piecewise graphs | 1.003 s | 29.03 ms | 29.66 ms | 34.45 tok/s |

Graphs improve decode by 8.46x and bring the first tiered tracer bullet to
within about 7% of the 37.06 tok/s native-offload baseline. The result remains
below the project target; the next execution optimization is independent
hot/cold streams with truly separate workspaces, followed by route tracing and
placement rather than additional cache-tier work.

After the graphed 4K request, allocator-retained prefill buffers reduced
physical free HBM to about 1.52 GiB per GPU. As with the host-cache scenario,
the startup reserve audit is not yet a worst-request high-water guarantee.
