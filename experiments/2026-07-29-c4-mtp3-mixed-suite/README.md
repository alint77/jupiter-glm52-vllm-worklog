# c4/MTP3 mixed-domain throughput

The current c4 serving path was measured on the original 18-prompt MTP suite:
three prompts each for Python, PyTorch, CUDA C++, math, email, and technical
explanation. Each request generated exactly 256 tokens at temperature zero.
The suite ran as one workload at request concurrency four, with one warmup and
two measured repetitions.

## Configuration

- GLM-5.2 AutoRound W4G64 plus the grafted MTP head
- one GH200 Booster node, TP4/EP4, DCP4, Model Runner V2
- MTP depth 3 and `max_num_seqs=4`
- current `hybrid-p0.5` placement, 7 GB HBM reserve
- full/piecewise CUDA graphs for verification sizes 4, 8, 12, and 16
- tight-shared-memory Marlin hot/cold overlap enabled
- staged ExaFlash checkpoint

## Result

| Metric | Repeat 1 | Repeat 2 | Mean |
| --- | ---: | ---: | ---: |
| Aggregate output throughput | 179.96 tok/s | 183.23 tok/s | **181.60 tok/s** |
| Decode-interval aggregate | 181.05 tok/s | 184.40 tok/s | **182.73 tok/s** |
| Per-request TPOT | 18.702 ms | 18.672 ms | **18.687 ms** |
| TTFT | 537.4 ms | 463.4 ms | **500.4 ms** |
| MTP draft acceptance | 70.13% | 68.25% | **69.19%** |
| Accepted tokens/target step | 3.104 | 3.047 | **3.076** |

Both repetitions completed all 18 requests with no failures and produced all
4,608 requested output tokens. Aggregate output throughput is the primary
number: it is total generated tokens divided by complete benchmark wall time,
including the short prefills and transitions between request waves. The
decode-interval value excludes time before the first output token.

Per-domain numbers below are mean per-request stream rates across the two
repetitions. They are not additive aggregate throughput:

| Domain | Decode tok/s | TPOT | TTFT |
| --- | ---: | ---: | ---: |
| Python | 56.68 | 17.731 ms | 530.9 ms |
| PyTorch | 51.06 | 19.660 ms | 592.0 ms |
| CUDA C++ | 53.23 | 19.133 ms | 489.8 ms |
| Math | 53.91 | 18.741 ms | 471.4 ms |
| Email | 48.49 | 20.683 ms | 525.3 ms |
| Explanation | 62.75 | 16.175 ms | 392.9 ms |

The two aggregate measurements differ by 1.8%, so the **181.6 tok/s** result is
stable at this sample size. The older 173.32 tok/s c4 result used a different
24-prompt technical suite and predates the shared-memory fix, so its 4.8%
numerical difference is context only, not a controlled before/after comparison.

## Artifacts

- `prompts.jsonl`: exact combined copy of the original six category files
- `mixed-c4-r{1,2}.json`: detailed benchmark results
- `summary.json`: aggregate and per-domain analysis
- `job.sbatch`, `analyze.py`: reproducible launcher and analysis

Jobs `1093991` and `1094280` ran on `jpbo-041-26` and `jpbo-062-11`.
The first populated the fresh compile cache and timed out before serving; the
second reused it and completed in 9:51.
