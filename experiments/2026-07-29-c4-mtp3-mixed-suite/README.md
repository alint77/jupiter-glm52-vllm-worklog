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

| Metric | Repeat 1 | Repeat 2 | Mean | rep spread |
| --- | ---: | ---: | ---: | ---: |
| **Per-request TPOT** | 18.702 ms | 18.672 ms | **18.687 ms** | **0.16%** |
| Aggregate output throughput | 179.96 tok/s | 183.23 tok/s | 181.60 tok/s | 1.8% |
| Decode-interval aggregate | 181.05 tok/s | 184.40 tok/s | 182.73 tok/s | 1.8% |
| TTFT | 537.4 ms | 463.4 ms | 500.4 ms | 15.9% |
| MTP draft acceptance | 70.13% | 68.25% | 69.19% | 1.9 pt |
| Accepted tokens/target step | 3.104 | 3.047 | 3.076 | 1.9% |

Both repetitions completed all 18 requests with no failures and produced all
4,608 requested output tokens.

**Quote TPOT, not the aggregate.** TPOT reproduces to 0.16% while the aggregate
reproduces to 1.8% - an order of magnitude worse - and the 1.8% is almost
entirely acceptance-rate noise: draft acceptance moved 1.9 points between the
two repetitions, which by itself moves throughput about 1.5%. The aggregate is
also computed over a 25-second run that is only 4.5 waves of four concurrent
requests, so it carries ramp-up and drain in the denominator. It answers "what
would this feel like" and is the right *headline* for a user; it is the wrong
metric to compare two builds with. For controlled before/after work use the
no-MTP protocol from
[the shared-memory phase](../2026-07-29-marlin-smem-monopoly/README.md), which
removes acceptance from the measurement entirely and reproduces to +-0.1 ms.

The older 173.32 tok/s c4 result used a different 24-prompt technical suite and
predates the shared-memory fix, so its 4.8% numerical difference is context only,
not a controlled before/after comparison.

## The per-domain table was measuring submission order

An earlier version of this report published per-domain decode rates and read the
spread between them as a domain effect. It is not. `vllm bench serve` runs with
`--disable-shuffle`, so **file order is submission order**, and `prompts.jsonl`
groups the domains in blocks of three. With concurrency four over 18 requests
that makes domain an alias for wave position, and the last requests submitted
finish with fewer co-running requests:

| grouped-order arm | submission indices | draining requests | mean TPOT |
| --- | --- | ---: | ---: |
| python | 0, 1, 2 | 0 | 17.731 ms |
| pytorch | 3, 4, 5 | 0 | 19.660 ms |
| cuda-cpp | 6, 7, 8 | 0 | 19.133 ms |
| math | 9, 10, 11 | 0 | 18.741 ms |
| email | 12, 13, 14 | 0 | 20.683 ms |
| explanation | 15, 16, 17 | **6 of 6** | **16.175 ms** |

Every `explanation` request is a draining request, and no other domain has one.
Measured directly, drain position is worth **15.7%**: 16.175 ms mean TPOT for the
six draining requests against 19.190 ms for the 30 steady ones. So
`explanation`'s apparent 29% advantage over `email` is batch occupancy, not
prose being easier to draft. `analyze.py` now emits `submission_indices`,
`draining_requests` and a `steady_vs_draining` block so this cannot be read
wrong again.

Per-domain TTFT is worse still: at concurrency four with 14 requests queued
behind, TTFT is dominated by queue position and should not be reported per domain
at all.

`prompts-interleaved.jsonl` reorders the same 18 prompts round-robin
(python, pytorch, cuda-cpp, math, email, explanation, repeat), so each domain
gets one early, one middle and one late slot and the drain penalty spreads
evenly. The interleaved arm is the one to read for domain effects.

`analyze.py` also no longer averages `1/tpot` for the rate column. Averaging
reciprocals biases a rate upward and made the rate and latency columns disagree
by about 0.5%; the rate is now derived from the mean TPOT.

## Artifacts

- `prompts.jsonl`: exact combined copy of the original six category files,
  grouped by domain (submission order equals domain)
- `prompts-interleaved.jsonl`: same 18 prompts, round-robin order
- `mixed-c4-r{1,2}.json`: grouped-order results
- `summary-mixed-c4.json`: aggregate, per-domain and steady-versus-draining
  analysis for the grouped-order arm
- `job.sbatch`, `analyze.py`: reproducible launcher and analysis. `PROMPT_FILE`
  and `RESULT_TAG` select the arm; the defaults are the interleaved one.

```bash
sbatch agent_space/experiments/2026-07-29-c4-mtp3-mixed-suite/job.sbatch
PROMPT_FILE=prompts.jsonl RESULT_TAG=mixed-c4 \
  sbatch agent_space/experiments/2026-07-29-c4-mtp3-mixed-suite/job.sbatch
```

Jobs `1093991` and `1094280` ran the grouped-order arm on `jpbo-041-26` and
`jpbo-062-11`. The first populated the fresh compile cache and timed out before
serving; the second reused it and completed in 9:51. Compile caches now live
under `/e/project1/.../marlin-caches` rather than `/e/scratch`, which ran out of
**inodes** (not space) and killed three jobs during the previous phase.
