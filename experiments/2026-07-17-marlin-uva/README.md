# GLM-shaped Marlin over Grace UVA

Phase 0 comparison of HBM-resident and final-destination pinned Grace/UVA
W4A16 weights on job `956247`, GPU 0, local Grace NUMA node 0.

The benchmark uses symmetric INT4, group size 128, FP16 activations, `M=1`,
and the native vLLM Marlin GEMM. HBM and Grace launches are interleaved with
alternating order to limit clock and ordering bias. Each row is the median of
300 timed groups; all HBM and Grace outputs match exactly.

## Single 6,144 x 2,048 projection sweep

| Distinct experts | HBM us/expert | Grace UVA us/expert | Grace / HBM |
|---:|---:|---:|---:|
| 1 | 74.848 | 77.632 | 1.037 |
| 2 | 56.680 | 57.648 | 1.017 |
| 4 | 45.340 | 45.712 | 1.008 |
| 8 | 39.146 | 39.026 | 0.997 |

## Complete expert matrix shapes, eight distinct experts

| Projection | Shape M x K x N | HBM us/expert | Grace UVA us/expert | Grace / HBM |
|---|---:|---:|---:|---:|
| Combined gate/up | 1 x 6,144 x 4,096 | 39.224 | 43.546 | 1.110 |
| Down | 1 x 2,048 x 6,144 | 44.470 | 43.908 | 0.987 |
| Sequential sum | both | 83.694 | 87.454 | 1.045 |

The complete-expert estimate is therefore about a 4.5% local kernel penalty
for pinned Grace UVA in this microbenchmark. Absolute latency moves with GPU
clocks, so the alternating within-run ratio is the useful result. This is not
an end-to-end MoE number: native fused dispatch, activation, routing, and
collectives are absent.

The largest pinned allocation sampled 192 Linux 64 KiB pages. All were on
NUMA node 0 before and after the Marlin run. Unlike the pageable experiment,
no sampled page migrated to paired HBM node 4. This validates pinned UVA as the
current cold-weight contingency and justifies continuing with destination-aware
expert bundles while the pageable LPDDR path remains disabled.

Reproduce from the vLLM checkout with:

```bash
srun --jobid=956247 --nodes=1 --ntasks=1 --gres=gpu:1 --overlap \
  --cpu-bind=map_cpu:0 bash -lc \
  'source agent_space/jupiter-env.sh && \
   .venv/bin/python agent_space/benchmarks/marlin_grace_uva.py \
   --experts 8 --groups 300 --k 6144 --n 4096'
```
