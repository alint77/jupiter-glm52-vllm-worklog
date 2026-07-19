# V2 runner at c1/q4

This experiment qualifies the V2 model runner for the normal single-request,
four-token verification workload. It uses GLM-5.2 W4A16, TP4/EP4, MTP3,
tiered MoE with the `0 / 75 / 0` frequency placement, NUMA binding, FP8 MLA
KV, full-and-piecewise CUDA graphs, and source `6490e584f`.

## Apparent startup hang

Job `977479` was killed by its 40-minute readiness limit after the target and
MTP `torch.compile` messages. It had not deadlocked: all ranks were still
running first-use Inductor autotuning for the MTP dense matrix shapes. Its
job-local FlashInfer and DeepGEMM caches also caused avoidable JIT work.

Jobs `977807`-`977809` reused the partially populated VLLM cache and the shared
FlashInfer cache. All three graph modes reached readiness and returned the
same semantic completion prefix (`" Paris. Distance from Paris to Lyon is"`).
The reusable scripts now seed from the completed V2 cache and use the shared
TRT-LLM cache.

| Graph mode | End-to-end tok/s | Mean TPOT | Decode tok/s | Acceptance length |
| --- | ---: | ---: | ---: | ---: |
| Full + piecewise | **15.58** | **8.342 ms** | **119.88** | 3.146 |
| Piecewise | 13.36 | 18.934 ms | 52.82 | 3.266 |
| Eager | 7.25 | 80.806 ms | 12.38 | 3.096 |

This four-prompt screen includes first-use JIT and is only suitable for choosing
the graph mode. Full graphs are required for useful V2 decode performance.

## Warmed realistic qualification

Job `977846` runs the same 24 Python, PyTorch, machine-learning, and math
prompts used by the V1 regression gate. Prefix caching is disabled. One full
24-prompt pass is excluded, followed by two saved repetitions of 24 requests
with 256 forced output tokens at concurrency one.

| Runner | End-to-end tok/s | Mean TPOT | Decode tok/s | Mean TTFT | Acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 full graphs | **98.03** | **9.252 ms** | **108.08** | **252.0 ms** | **2.895** |
| V2 full graphs | 95.44 | 9.427 ms | 106.08 | 278.7 ms | 2.853 |

V2 is 1.85% slower in steady decode and 2.65% slower end to end. Both V2
repetitions completed 24/24 requests with no errors, every response produced
the requested 256 tokens, and sampled Python/PyTorch/ML/math responses were
coherent. The small throughput loss tracks a slightly lower acceptance length
and does not justify changing the c1/q4 default.

The V2 runner remains qualified for c4/DCP4, where complete MTP graphs produced
the earlier 20.8% gain. For c1/q4, retain V1 and preserve the populated
`VLLM_CACHE_ROOT`, shared FlashInfer cache, and shared TRT-LLM cache for any
future retest. The 370 GiB checkpoint read still takes roughly 11-18 minutes
from GPFS and is separate from JIT startup.

Status: complete; V2 c1/q4 is functional but rejected on performance.
