# C4 MTP-depth and residency-edge sweep

Goal: compare MTP1, MTP2, and MTP3 at concurrency four on the qualified
TP4/EP4/DCP4 V2-runner path, while minimizing Grace expert offload.

All three configurations start at the tiered runtime's minimum legal HBM
reserve: 7 decimal GB/rank. Relative to the prior 10 GB c4 runs, this makes
about 3 GB/rank more HBM available to routed experts. The domain-trained
per-expert profile remains the ordering seed; the planner deterministically
promotes additional cold experts into the newly available capacity.

The header-only physical plan resolves to 3,023 hot and 1,777 cold expert
slots per rank. Cold expert storage is 34,587,883,400 bytes/rank, down
2,997,486,800 bytes from the prior 2,869-hot layout. The c4/DCP4 KV plan is
unchanged at 6,253 blocks and 21,901,707,776 HBM bytes/rank.

Each job runs:

1. semantic and exact-400K SHA correctness gates;
2. two cache-cold repetitions of the 24-prompt Python/PyTorch/ML/math suite at
   concurrency four;
3. two matched 4K-input, 256-output c4 repetitions;
4. a prefix-primed 395,904-input c4 decode measurement; and
5. an eight-step, four-rank PyTorch profile of that full-context c4 decode.

Raw profiler traces are written to scratch and their paths are recorded with
the final results.

## Jobs

| Job | MTP depth | Initial reserve | State |
| --- | ---: | ---: | --- |
| 1022345 | 1 | 7 GB | edge qualified; harness 404 |
| 1022346 | 2 | 7 GB | edge qualified; harness 404 |
| 1022347 | 3 | 7 GB | edge qualified; harness 404 |
| 1022400 | 1 | 7 GB | complete |
| 1022401 | 2 | 7 GB | complete |
| 1022402 | 3 | 7 GB | complete |

The first jobs proved loading, compilation, graph capture, and the physical
HBM edge, then exited because `/reset_prefix_cache` is a development endpoint
and was not registered. The retries set `VLLM_SERVER_DEV_MODE=1`; servers
remain bound to loopback.

The retries used source `fcd9fb65c`, the V2 runner, full graphs, `ag_rs`,
TP4/EP4/DCP4, and the same placement, seeds, and requests. Model weights use
64.38 GiB/rank, KV uses 18.55 GiB/rank, and graphs use 1.18--1.24 GiB/rank.
The runtime reports only 5.6--5.7 GiB/rank between the current KV allocation
and full physical HBM use. All three depths therefore qualify at the minimum
legal tiered reserve without an OOM.

## Performance

All requests completed without failure. Every depth produced the expected
semantic completion and the exact 400K golden SHA
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`.

| Measurement | MTP1 | MTP2 | MTP3 |
| --- | ---: | ---: | ---: |
| Realistic c4 first-use output tok/s | 100.08 | 101.46 | **107.08** |
| Realistic c4 warmed output tok/s | 163.68 | 158.55 | **179.47** |
| Realistic mean TPOT, two-run mean (ms) | 24.39 | 23.66 | **20.79** |
| Realistic effective aggregate decode (tok/s) | 194.80 | 194.40 | **228.33** |
| 4K c4 output tok/s, two-run mean | 92.10 | 90.86 | **98.54** |
| 4K c4 effective aggregate decode (tok/s) | 163.83 | 170.88 | **201.38** |
| 396K c4 median ITL (ms) | **45.02** | 61.47 | 60.10 |
| 396K c4 acceptance length | 1.925 | 2.451 | **2.665** |
| 396K c4 effective aggregate decode (tok/s) | 171.04 | 159.46 | **177.35** |

Effective aggregate decode is
`4 * acceptance_length / median_ITL_seconds`, matching the earlier c4
reporting convention. The first realistic pass pays first-use varied-prefill
cost; decode metrics are stable across both passes, while the second pass is
the warmed end-to-end result.

MTP3 is the clear choice for the realistic and 4K workloads. At 396K, MTP1
has the shortest raw step, but its lower acceptance leaves MTP3 3.7% ahead in
effective aggregate throughput. MTP2 does not provide a consistent middle
point: it is 4.3% ahead of MTP1 on the 4K effective metric, effectively tied
on realistic prompts, and 6.8% behind at 396K.

The fresh MTP3 4K effective result is 10.6% below the older 225.23 tok/s
number. This is not an apples-to-apples source regression: the old run used
the generic graft placement profile and a 10 GB reserve, while this sweep uses
the domain-trained profile and the 7 GB edge layout. The current realistic
suite is the primary result; the random 4K difference is a placement/config
regression to isolate if that synthetic workload matters.

## Trace result

Two of the eight profiled engine iterations contained the final 1,152-token
prefix tails. The remaining six iterations are homogeneous c4 full-graph
decode.
The analyzer now selects the dominant homogeneous geometry instead of the
first mixed-shape annotation; its output remains byte-identical on the older
homogeneous trace, apart from the newly recorded MTP-depth metadata.

| Trace metric per step | MTP1 | MTP2 | MTP3 |
| --- | ---: | ---: | ---: |
| Wall time (ms) | **45.12** | 63.63 | 62.43 |
| GPU union busy (ms) | 40.45 | 49.85 | 54.48 |
| GPU idle (ms) | 4.67 | 13.78 | 7.95 |
| Kernel launches | 4,579 | 4,694 | 4,794 |
| Routed W4 MoE activity (ms) | 28.04 | 36.54 | 42.44 |
| TP custom all-reduce activity (ms) | 7.13 | 9.03 | 9.23 |
| Sparse MLA activity (ms) | 2.08 | 2.34 | 2.65 |
| Hot/cold MoE overlap saved | 40.4% | 40.6% | 40.3% |

The MTP head itself is cheap: its FP8 block grows only from 0.22 to 0.65
ms/step from MTP1 to MTP3. Routed Marlin remains dominant. Every depth launches
one cold expert cluster in all 75 routed layers; cold C2C activity grows from
11.90 to 18.20 ms, while hot HBM activity grows from 16.14 to 24.24 ms.
The deeper batch improves hot-path roof efficiency from a 32--39% bound at
MTP1 to 46--53% at MTP3.

The main new anomaly is MTP2's 13.78 ms of idle time: its p99/max GPU gap is
95.8/686.5 us versus 38.0/245.8 us for MTP3. The critical-path analysis below
identifies the concrete graph-capture hole behind it and changes the priority
of the cold-path and communication work.

## Critical-path diagnosis

CUDA-runtime correlations identify one 4,436-kernel target graph per engine
step. The target graph is followed by the recurrent MTP proposer work before
the next target graph starts. Averaging the four ranks and four interior
cycles gives:

| Critical-path component (ms/step) | MTP1 | MTP2 | MTP3 |
| --- | ---: | ---: | ---: |
| Target-start to next target-start | 45.12 | 63.62 | 62.49 |
| Target graph span | 43.30 | 50.48 | 55.70 |
| Idle inside target graph | 3.80 | 3.79 | 3.79 |
| Post-target proposer/orchestration span | 1.82 | 13.14 | 6.79 |
| Post-target GPU work | 0.95 | 3.15 | 2.63 |
| Post-target GPU idle | 0.87 | **9.99** | 4.16 |

This is not a return to thousands of host kernel launches. MTP3 replays one
large target graph and three small proposer graphs. Its remaining host-side
opportunity is the 4.16 ms of bubbles while sequencing those recurrent draft
graphs and the next engine step. The invariant 3.79 ms inside the target graph
is graph-node/dependency overhead, not CPU launch latency for 4,436 individual
kernels.

MTP2 has a configuration-specific capture hole. The server requested graph
sizes `[3, 6, 9, 12]`, which cover its three-token verification multiples but
omit the four-token draft batch at c4. The trace consequently contains about
70 `cudaLaunchKernel` and 20 `cudaLaunchKernelExC` calls per draft tail and
9.99 ms of post-target idle. MTP1's size list includes four, while MTP3's
`[4, 8, 12, 16]` does too. Future runs must capture the union of verification
sizes and draft batch sizes.

For MTP3, the target graph is the dominant 55.70 ms of the 62.49 ms cycle.
Routed Marlin occupies a 25.35 ms union, or 40.6% of the outer step, and is
alone on the GPU critical path for 20.23 ms. The hot path controls 99.6% of
the 1,200 observed rank/layer expert clusters. Cold C2C work is active in
every routed layer but extends the observed hot/cold union by only 1.10 ms in
total. Eliminating cold work therefore has only a 1.8% direct step-time
ceiling unless it also relieves contention that slows the hot kernels.

The apparent TP communication problem is primarily an expert-balance problem.
Across the 166 custom reductions per MTP3 step, ranks enter a reduction
99.4 us apart on average but finish only 3.1 us apart. Comparing each
reduction's rank durations attributes about 8.36 ms/step to early-rank
waiting. Independently, summing the per-layer difference between the slowest
rank's routed span and the four-rank mean gives a 7.80 ms/step imbalance
proxy. The close match shows that the reductions are exposing the
layer-by-layer slowest expert rank; their low analytical 6.8% link efficiency
does not by itself establish a raw NVLink bandwidth bottleneck.

At 396K, MTP depth is limited by acceptance as much as latency:

| Depth | Step (ms) | Accepted tokens/sequence | Position acceptance |
| --- | ---: | ---: | --- |
| MTP1 | 45.12 | 1.925 | 92.5% |
| MTP2 | 63.63 | 2.451 | 90.6%, 54.5% |
| MTP3 | 62.43 | 2.665 | 90.8%, 52.0%, 23.7% |

MTP3 accepts 38.4% more tokens than MTP1 while its traced step is also 38.4%
longer, leaving almost no long-context speculative gain. On the realistic
suite the third position accepts 43.6%, so MTP3 has enough useful work to win.
This domain/context sensitivity explains why a kernel-only optimization
cannot guarantee the same aggregate gain on every workload.

The resulting priorities are:

1. include draft batch sizes in graph capture, which should remove the MTP2
   anomaly;
2. balance routed expert work per layer/rank, then remeasure custom
   all-reduce wait;
3. reduce the 4.16 ms MTP3 recurrent-proposer/engine bubbles; and
4. pursue target-graph fusion only after the first three, since cold removal
   alone has little direct critical-path leverage.

Raw Perfetto traces:

- MTP1:
  `/e/scratch/profound/naeimitabiei1/glm52-c4-mtp1-edge-profile-1022400`
- MTP2:
  `/e/scratch/profound/naeimitabiei1/glm52-c4-mtp2-edge-profile-1022401`
- MTP3:
  `/e/scratch/profound/naeimitabiei1/glm52-c4-mtp3-edge-profile-1022402`

Each directory contains one compressed trace per rank. The checked-in
`c4-mtp{1,2,3}-edge-summary.json`, kernel inventories, and roofline plots are
the compact derived artifacts.
