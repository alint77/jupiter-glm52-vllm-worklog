# Placement trace comparison

Jobs `1048432` and `1048433` profiled the c4/DCP4 control and the best new
profile, balanced owners plus frequency residency. Both use the AutoRound
W4G64 checkpoint, MTP3, identical graph shapes, and the same 24 realistic
prompts. Each trace contains eight steady engine iterations, 600 recognized
routed-layer executions per rank, and exactly 37,316 GPU events per rank.

The uncompressed Perfetto traces are in:

- `/e/scratch/profound/naeimitabiei1/claude-placement-profiles/control-1048432`
- `/e/scratch/profound/naeimitabiei1/claude-placement-profiles/balanced-owners-frequency-1048433`

## Result

| Rank-mean trace metric | Control | New profile | Delta |
|---|---:|---:|---:|
| GPU span | 447.97 ms | 512.86 ms | +14.49% |
| GPU busy | 422.24 ms | 488.01 ms | +15.58% |
| GPU idle share | 5.74% | 4.84% | -0.90 pp |
| Marlin kernel time | 339.56 ms | 425.45 ms | +25.30% |
| MoE activation/sum time | 59.56 ms | 76.43 ms | +28.32% |
| Collective time | 92.21 ms | 101.97 ms | +10.59% |
| Routing time | 11.53 ms | 11.55 ms | +0.25% |

The candidate is busier, not more launch-idle. Marlin accounts for the
regression; attention, other GEMMs, routing, and memory operations are nearly
unchanged.

The two-tier Marlin decomposition is more specific:

| Eight-step rank mean | Control | New profile | Delta |
|---|---:|---:|---:|
| Hot W13 | 132.85 ms | 173.67 ms | +30.72% |
| Hot W2 | 57.14 ms | 77.16 ms | +35.04% |
| Cold W13 | 62.76 ms | 56.92 ms | -9.32% |
| Cold W2 | 84.43 ms | 115.41 ms | +36.69% |
| Combined layer span | 213.21 ms | 268.58 ms | +25.97% |
| Hot/cold overlap saving | 36.77% | 36.53% | -0.24 pp |
| Cold chain finishes first | 91.83% | 96.08% | +4.25 pp |

The offline objective minimized the maximum per-rank number of cold routes.
That proxy assumes the cold kernel is the critical tier. The trace disproves
the assumption for this workload: the cold chain finishes first in almost
every routed layer, while the new profile increases hot-chain time by 32%.
More HBM route coverage is therefore not equivalent to lower layer latency.

## Next placement objective

Keep the current owner map initially and replace the cold-count proxy with a
calibrated two-chain cost:

```text
layer cost = max(cost_hot(route shape), cost_cold(route shape)) + join cost
```

Fit separate W13 and W2 costs from trace data, including active experts and
tokens per expert. Select HBM residency against the maximum hot/cold chain,
not cold coverage alone. Validate chronologically on the captured Claude
routes, then test only the Pareto candidates that preserve the short-prompt
control while improving predicted Claude latency. Owner rebalancing and tail
swaps remain secondary because their held-out gains were negligible.

`placement-trace-comparison.json` contains per-rank kernel categories, top
kernels, idle gaps, and hot/cold chain measurements. It is reproducible with
`benchmarks/compare_placement_traces.py`.
