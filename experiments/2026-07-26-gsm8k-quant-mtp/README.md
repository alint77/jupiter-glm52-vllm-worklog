# GSM8K quantization and MTP comparison

Goal: compare downstream accuracy and end-to-end output throughput for four
matched c1/TP4/EP4 configurations:

| Label | Checkpoint | Speculation |
|---|---|---|
| `w4-target` | GLM-5.2 compressed-tensors W4G128 | disabled |
| `w4-mtp3` | GLM-5.2 compressed-tensors W4G128 | MTP3 |
| `autoround-target` | AutoRound W4G64 | disabled |
| `autoround-mtp3` | AutoRound W4G64 | MTP3 |

The evaluation uses the first 256 GSM8K test questions, the evaluator's
standard five-shot completion prompt, greedy decoding, natural EOS, and a
256-token output limit. Each server gets an unmeasured eight-question warmup
followed by two measured repetitions. The headline TPS is total generated
tokens divided by evaluation wall time; the final comparison averages this
value across the two repetitions.

All four jobs use the same source revision and serving policy. Each checkpoint
uses its corresponding version of the same domain-mixed expert ranking. MTP is
the only within-checkpoint difference.

Dataset:

- source: `openai/grade-school-math`
- train SHA-256:
  `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`
- test SHA-256:
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`

## Results

All four jobs completed with exit code zero on source `4e563c699`. Every
measured request returned a scorable answer; invalid rate was zero throughout.

| Configuration | Accuracy r1 / r2 / mean | Output tok/s r1 / r2 / mean | MTP acceptance | Minimum free HBM/rank |
|---|---:|---:|---:|---:|
| W4 target | 94.53 / 92.58 / 93.55% | 41.52 / 41.52 / **41.52** | - | 12.76 GiB |
| W4 MTP3 | 94.53 / 94.14 / 94.34% | 79.72 / 80.38 / **80.05** | 76.82% | 7.35 GiB |
| AutoRound target | 95.31 / 95.70 / 95.51% | 42.41 / 42.39 / **42.40** | - | 12.13 GiB |
| AutoRound MTP3 | 96.09 / 95.31 / 95.70% | 77.87 / 78.31 / **78.09** | 77.76% | 10.10 GiB |

The throughput conclusions are clean:

- MTP3 improves W4 end-to-end output throughput by 92.78%.
- MTP3 improves AutoRound throughput by 84.19%.
- AutoRound is 2.10% faster than W4 target-only, but 2.45% slower with MTP3.
- AutoRound's MTP acceptance is 0.94 percentage points higher, with an average
  accepted/emitted span of 3.333 tokens versus W4's 3.305.

Accuracy is not decisive at this sample size. AutoRound's two-run mean is 1.95
points above W4 without MTP and 1.37 points above it with MTP3, but the same W4
target varied by 1.95 points across identical repetitions. These repetitions
reuse the same 256 questions and therefore are a stability check, not 512
independent samples. Treat the checkpoints as tied unless a full 1,319-question
run reproduces the difference.

HBM values are active-generation snapshots from `nvidia-smi`; the table reports
the least free memory of the four ranks. Model allocations are stable during
decode, and no OOM or memory warning occurred. MTP3 costs 5.41 GiB of headroom
for W4 and 2.03 GiB for AutoRound under these matched settings.

Jobs:

- `1046095`: W4 target, 38:44
- `1046096`: W4 MTP3, 31:44
- `1046097`: AutoRound target, 41:37
- `1046098`: AutoRound MTP3, 33:39

Raw artifacts include both result JSON files per configuration, server and
Slurm logs, before/after Prometheus metrics, and load/active HBM snapshots.
