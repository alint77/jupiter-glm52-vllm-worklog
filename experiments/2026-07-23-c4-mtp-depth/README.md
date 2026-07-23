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
95.8/686.5 us versus 38.0/245.8 us for MTP3. That excess launch/arrival tail
explains why MTP3 completes a larger verification batch slightly faster.
The next optimization targets are therefore:

1. remove the MTP2 graph/collective arrival gaps;
2. reduce the always-active cold-expert layer count or improve hot/cold
   overlap under the c4 domain routing distribution; and
3. reduce the 7--9 ms TP all-reduce chain, whose modeled link efficiency is
   only 4.4--6.8%.

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
