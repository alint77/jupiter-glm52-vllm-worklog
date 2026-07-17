# Phase 7 end-to-end tuning

Phase 7 closes the remaining collective, populated-400K, and physical-memory
measurements. Earlier phases already measured Marlin/UVA kernels, hot/cold
stream overlap, workspace sharing, graph boundaries, cache tiers, and
profiled versus unprofiled placement.

## Decode collectives

The communicator benchmark now accepts `--hidden-size`, allowing the exact
GLM decode tensor: one token by 6,144 BF16 elements, or 12 KiB. TP4 results use
CUDA graphs, 100 warmups, and 1,000 trials with ten reductions per replay.

| Backend | 12 KiB latency |
| --- | ---: |
| Custom one-stage | 4.058 us |
| Custom two-stage | 6.755 us |
| PyNCCL | 11.916 us |
| PyNCCL symmetric-copy | 13.777 us |

The server already selects custom one-stage before PyNCCL. Each decode step
has 157 of these reductions: one embedding reduction, 78 attention output
reductions, and 78 MLP output reductions. Their serialized measured cost is
about 0.637 ms/token, versus 1.871 ms with PyNCCL. Logit projection adds one
TP vocabulary all-gather. No collective implementation change is justified.

## Full-context result

All runs use the 400K HBM main cache, trace-derived owners, full/piecewise
graphs, one 4K/16 warmup, and one random 399,744-input/256-output request.
Decode rate is `1000 / TPOT`.

| Configuration | TTFT | TPOT | Decode | Min free HBM |
| --- | ---: | ---: | ---: | ---: |
| Native CPU offload baseline | 120.853 s | 26.617 ms | 37.57 tok/s | n/a |
| Tiered linear, 7 GB reserve | 118.385 s | 27.345 ms | 36.57 tok/s | 2.52 GiB |
| Profiled, 7 GB reserve, seed 13 | 106.182 s | 24.238 ms | 41.26 tok/s | 1,439 MiB |
| Profiled, 10 GB reserve, seed 13 | 106.999 s | 23.721 ms | 42.16 tok/s | 4,295 MiB |
| Profiled, 10 GB reserve, seed 14 | 107.020 s | 23.808 ms | 42.00 tok/s | 4,296 MiB |

The selected reserve-10 mean is 107.009 seconds TTFT, 23.765 ms TPOT, and
42.08 tok/s. The two TPOT samples differ by 0.37%. Relative to the native
baseline, prefill latency falls 11.46%, decode latency falls 10.72%, and decode
rate rises 12.00%. Relative to tiered linear placement, decode rate rises
15.06%.

Seven planned reserve GB left only about 1.4 GiB physically free after the
full request. The exact 10 GB plan moves 155 more expert slots per rank to
Grace: 3,022 stay hot and 1,778 stay cold. Model HBM falls from 65.66 to
62.87 GiB/rank, idle free HBM rises from about 6.18 to 8.97 GiB, and post-400K
free HBM is 4,295-4,302 MiB. This clears the v2 observed-memory gate without a
measured performance loss.

The reserve-10 semantic smoke exactly matches the earlier qualified output:
` Paris. Distance from Paris to Lyon is`.

## Status

Phase 7's measurement checklist is complete, including two populated-400K
requests with 256 warmed output tokens and separate TTFT reporting. The
100 tok/s project minimum is not met: the selected result is 42.08 tok/s.
The next useful step is a one-token CUDA critical-path trace to identify the
largest remaining kernel group before changing code; collectives account for
only about 2.7% of measured TPOT.

Raw JSON, the exact reserve-10 profile, optimizer report, memory samples,
launch scripts, and server/client logs are stored beside this report.
