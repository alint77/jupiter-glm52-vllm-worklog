# MTP3 c4 speed-of-light and GPU-idle analysis

## Scope and metric clarification

This report analyzes the four rank traces in:

```text
/e/scratch/profound/naeimitabiei1/glm52-c4-mtp3-edge-profile-1022402
```

The trace is steady concurrency-four decode at 395,904 prefix tokens,
MTP3, TP4/EP4/DCP4, `ag_rs`, full CUDA graphs, and the 7 GB tiered reserve.
It contains four homogeneous interior steps per rank.

Subtracting the current 179.47 tok/s from the earlier 225.23 tok/s result is
not a valid regression calculation. The current realistic number is
end-to-end output throughput, including prefill, request refill, and
batch-drain time. Under the earlier steady-decode convention it is
228.33 tok/s. The matched current 4K effective result is 201.38 tok/s, 10.6%
below the older one-shot 225.23 result, but this 396K trace cannot attribute a
4K-only difference. At its measured context, the trace corresponds to
177.35 tok/s from live median ITL and acceptance, consistent with the earlier
stable 173--179 tok/s long-context range.

The remaining issue is poor absolute scaling: four concurrent requests deliver
far less than four times the c1 rate. The trace explains that inefficiency.

## What SOL means here

Kernel durations, launch geometry, concurrency, and gaps are measured from the
PyTorch/CUPTI trace. The trace does not contain hardware performance counters.
FLOPs and minimum logical bytes are derived from model geometry and kernel
contracts, using these conservative GH200 ceilings:

| Resource | Ceiling |
| --- | ---: |
| HBM | 3.5 TB/s |
| BF16 tensor cores | 630 TFLOP/s |
| FP8 tensor cores | 1,260 TFLOP/s |
| Grace-to-Hopper C2C | 421 GB/s |
| Peer NVLink direction | 150 GB/s |

The reported SOL efficiency is
`achieved / min(compute peak, arithmetic intensity * bandwidth peak)`.
It is an analytical roofline estimate, not measured SM-active, tensor-active,
L2, or DRAM utilization. The profiler's launch-occupancy field is also launch
geometry, not a hardware-counter measurement.

## Whole-step utilization

| Measurement | Per step | Share of wall |
| --- | ---: | ---: |
| Target-start to target-start wall | 62.49 ms | 100.0% |
| GPU union busy | 54.48 ms | 87.3% |
| GPU idle | 7.95 ms | 12.7% |
| Cumulative CUDA activity | 77.58 ms | 124.2% |
| Kernels and GPU operations | 4,794 | n/a |

Concurrent hot/cold streams make cumulative activity exceed wall time. Of all
GPU operations, 3,067 per step, or 64.0%, last less than 5 us. They contribute
only 6.64 ms of activity but create a large graph-node scheduling surface.

The GPU is therefore not predominantly empty. It is busy 87% of the time, but
substantial busy time is low-SOL small-batch work or synchronization waiting.

## Kernel SOL

Times are cumulative activity and may overlap. The routed tier row uses the
observed hot/cold union because its two streams overlap.

| Kernel family | Time/step | Applicable SOL efficiency | Finding |
| --- | ---: | ---: | --- |
| Routed target W4 Marlin | 25.35 ms union | hot 46--53% HBM | Dominant path |
| Target W4 Machete | 3.65 ms | 14.12% | 312 skinny GEMMs |
| Sparse MLA split/combine | 2.65 ms | 5.38% | Lowest major kernel SOL |
| DSA full-context FP8 scan | 1.06 ms | 21.03% | Compute-roof limited model |
| Target QKV-A W4 Marlin | 0.91 ms | 20.26% | Dense weights amortize well |
| DSA stable top-k | 0.83 ms | 5.52% byte roof | Selection work not FLOP-modeled |
| MLA BF16 contractions | 0.70 ms | 25.73% | Small memory-bound GEMMs |
| Vocabulary projection | 0.57 ms | 95.71% | Already at HBM SOL |
| Target MoE router | 0.33 ms | 20.25% | Small memory-bound GEMMs |
| Target DSA Wq | 0.17 ms | 60.88% | Healthy |
| MTP FP8 routed experts | 0.38 ms | 68.80% | Healthy and small |
| MTP FP8 dense linears | 0.16 ms | 37.22% | Small total leverage |
| MTP BF16 projection | 0.15 ms | 83.69% | Healthy |

The target dense path is not responsible for the lost scaling. Using the
earlier c1 q4 trace as a geometry reference rather than a source A/B, QKV-A
remains near 0.9 ms while processing four times the target tokens, and
Machete falls from 4.65 to 3.65 ms. Dense weight reads are being amortized.

Routed experts do not obtain the same amortization. A q16 c4 verification
step creates 2,400 expert assignments per rank across diverse experts. The
hot/cold routed union is 25.35 ms and routed Marlin is the only active role
for 20.23 ms, or 32.4% of the complete outer step.

Cold C2C is not setting this critical path. The hot path finishes last in
99.6% of the 1,200 measured rank/layer clusters. Removing all observed cold
work would reduce the hot/cold union by only 1.10 ms directly. The useful
target is hot Marlin efficiency and expert/rank balance.

## Synchronization presented as communication

Custom TP reductions occupy 9.23 ms, or 14.8% of wall, and their payload-only
link model reports 6.79% of one NVLink direction. That number does not imply
that the link is saturated inefficiently.

Across 166 reductions per step:

- ranks enter a reduction 99.4 us apart on average;
- ranks finish only 3.1 us apart on average;
- comparing each call's rank durations gives 8.36 ms/step of early-rank
  waiting; and
- an independent layer-by-layer routed-rank imbalance proxy is 7.80 ms/step.

The close agreement shows that collectives are barriers exposing whichever EP
rank is slowest in each layer. Expert balance must be fixed before treating
the 9.23 ms as a collective bandwidth problem. The 8.36 and 7.80 ms values
describe the same loss and must not be added.

## Exact GPU-idle decomposition

CUDA-runtime correlations identify one 4,436-node target graph followed by
three small MTP proposer graphs.

| Region | Wall | GPU idle | Idle share |
| --- | ---: | ---: | ---: |
| Target graph | 55.70 ms | 3.79 ms | 6.8% |
| Draft graph 1 | 0.88 ms | 0.06 ms | 6.8% |
| Draft graph 2 | 0.73 ms | 0.03 ms | 4.2% |
| Draft graph 3 | 0.73 ms | 0.03 ms | 3.5% |
| Target to draft 1 | 0.35 ms | 0.04 ms | 12.8% |
| Draft 1 to draft 2 | 0.84 ms | 0.81 ms | 96.9% |
| Draft 2 to draft 3 | 0.77 ms | 0.74 ms | 96.9% |
| Draft 3 to next target | 2.50 ms | 2.45 ms | 97.8% |

The target graph contains 3,326 idle intervals per step averaging about
1.14 us. Every target-graph gap is below 20 us. These 3.79 ms are accumulated
CUDA-graph node/dependency scheduling gaps, not thousands of CPU-issued kernel
launches.

The 4.04 ms between graphs is different and directly actionable. Between each
pair of draft graphs, the host issues about 11 eager kernels and three
asynchronous copies. Between the final draft and next target it issues about
21 eager kernels and eight copies, alongside repeated slicing, copying,
pinned-memory, `repeat_interleave`, `cumsum`, and batch-construction work.
This is recurrent proposer and engine orchestration.

Idle duration distribution:

| Individual gap | Idle/step | Share of all idle |
| --- | ---: | ---: |
| Below 1 us | 1.28 ms | 16.1% |
| 1--5 us | 2.17 ms | 27.3% |
| 5--20 us | 0.60 ms | 7.6% |
| 20--100 us | 2.63 ms | 33.1% |
| At least 100 us | 1.26 ms | 15.9% |

There is no single multi-millisecond GPU stall; the largest individual gap is
246 us. The loss is the accumulation of thousands of sub-5-us graph gaps and
roughly one hundred host-orchestration gaps between graph replays.

Even eliminating all 7.95 ms of GPU idle would improve the 62.49 ms cycle by
only 14.6%. Eliminating only the actionable 4.04 ms between graphs has a 6.9%
ceiling. Idle time is important, but cannot by itself explain or recover
four-times c1 scaling.

## Why aggregate scaling is low

The limiting picture is:

1. Dense q16 work amortizes successfully.
2. Diverse routed experts do not amortize like dense weights; routed Marlin
   owns 40.6% of wall and reaches only about half of its modeled HBM SOL.
3. Layer-local EP imbalance turns the custom reductions into approximately
   8 ms of rank waiting.
4. GPU idle costs another 7.95 ms, split nearly evenly between target-graph
   microgaps and host work between proposer graphs.
5. At 396K, only 2.665 tokens/sequence are accepted from a maximum four. The
   third draft position accepts only 23.7%.

The GPU is neither primarily filesystem-I/O bound nor broadly CPU-launch
bound. It is dominated by routed-expert execution and fine-grained EP
synchronization, with a smaller but clean proposer-orchestration opportunity.

For MTP3, the trace-driven optimization order is:

1. balance expert ownership/work across EP ranks per layer;
2. remove the 4.04 ms between proposer/target graphs;
3. improve the hot routed-Marlin q16 path;
4. optimize sparse MLA and the 3.79 ms of target graph microgaps; and
5. leave the MTP FP8 head and vocabulary GEMM alone.
