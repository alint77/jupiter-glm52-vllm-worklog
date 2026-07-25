# P4 — what a CUDA-graph node actually costs

The critical-path review put ~7.6 ms/step in "small-kernel overhead": 4.65 ms of
glue execution plus ~3.78 ms of intra-target-graph idle spread over ~3,326 gaps
averaging ~1.14 us. The implied model is that each graph node carries a fixed
scheduling gap, so removing K nodes returns about K x 1.14 us.

Given that three earlier headline estimates in this project turned out to be
accounting errors, the model was measured before any fusion work was started.
Job 1042017, Booster, `benchmarks/kernels/benchmark_cudagraph_node_cost.py`.

## The model holds

**Marginal cost per node**, from replay time versus node count for a serial
chain of dependent elementwise kernels:

| kernel width | marginal us/node | fixed us |
| --- | ---: | ---: |
| 1,024 elements | **1.089** | −33.3 |
| 65,536 elements | **1.201** | +2.1 |

The trace's inferred ~1.14 us sits between the two measured slopes.

**Fusion control** — total work held constant, node count halved. This is the
proposal itself, in miniature:

| nodes | numel | total elements | replay us | vs previous |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 4,096 | 8,388,608 | 2,256.05 | — |
| 1,024 | 8,192 | 8,388,608 | 1,156.72 | **−48.7%** |
| 512 | 16,384 | 8,388,608 | 587.10 | −49.2% |
| 256 | 32,768 | 8,388,608 | 304.18 | −48.2% |

Halving 2,048 → 1,024 nodes returned **1.074 us per node removed**, and the
return is linear across three halvings with no sign of saturation.

**The cost is scheduling, not work.** Per-node gap against kernel width:

| numel | us/node | solo kernel us | **gap us** |
| ---: | ---: | ---: | ---: |
| 1,024 | 1.014 | 0.109 | 0.905 |
| 16,384 | 1.154 | 0.000 | 1.154 |
| 262,144 | 1.405 | 0.000 | 1.405 |
| 4,194,304 | 6.160 | 4.031 | 2.129 |

The gap is roughly constant at ~0.9–1.4 us for anything up to 262K elements —
which covers every glue kernel in the target graph, all of which run under 5 us.

So **removing K nodes saves about K x 1.07 us**, and the estimate can be
trusted.

## Honest sizing of the prize

Target-graph inventory (mean of 7 replays, rank 0, post-draft-sync trace):
**4,436 nodes, 70.674 ms cumulative kernel time**. Of those, **2,737 nodes (62%)
average under 5 us and contribute 5.921 ms — 8% of the time**.

Measured intra-graph idle is 3.78 ms over 4,436 nodes = **0.85 us of *exposed*
cost per node**, against 1.07 us in a fully serial chain. The difference is the
part hidden by the graph's five concurrent streams, so roughly 80% of node
scheduling sits on the critical path.

That fixes the ceiling: **eliminating every intra-graph gap would return 3.78 ms,
6.1% of the 62.43 ms step** — and only by deleting all 4,436 nodes, which is not
a thing that can happen.

Realistic slices, priced at 0.85 us per node removed:

| Slice | nodes removed | saving | share of step |
| --- | ---: | ---: | ---: |
| Fuse the two tiers' `moe_align` + `count_and_sort` + fill into single launches | ~225 | 0.19 ms | 0.3% |
| Single-tier MoE: merge hot/cold into one expert buffer, halving the whole per-layer epilogue (`moe_align`, `count_and_sort`, `act_and_mul`, `moe_sum_vec` at 150 → 75 each, the `add_` join, and 300 → 150 Marlin launches) | ~525 | 0.45 ms | 0.7% |
| Both, plus attention-layer glue (the eight ~78-count kernels) | ~900 | 0.77 ms | 1.2% |

**P4 is real, correctly modelled, and small.** A substantial fusion campaign
returns 1–1.5%. That is the honest number, and it is an order of magnitude below
the levers this project chased earlier — the difference being that this one is
measured rather than inferred.

## Recommendation

Do not start a fusion campaign for 1%. Two qualifications:

- If the **single-tier MoE merge** is built for another reason, it carries the
  ~0.45 ms node saving along for free. Note that P1b showed the two tiers
  already overlap at 81%, so merging them costs nothing in overlap — the
  argument for it is node count and the removal of the fork/join, not
  concurrency.
- The per-node figure is now a reusable constant. Any future change that adds
  or removes graph nodes can be priced at **~0.85 us/node exposed** without
  re-measuring, which makes it cheap to reject node-adding designs.

The remaining lever with real headroom is P3: the hot-slot frontier is flat past
~3,000 slots/rank, so HBM should buy KV and concurrency instead of experts.
