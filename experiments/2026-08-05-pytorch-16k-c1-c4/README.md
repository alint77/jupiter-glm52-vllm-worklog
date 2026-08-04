# PyTorch code generation, 16K prefill / 512 output, c1 and c4

Status: **Complete.** Measured against the live production server (job
`1238882`), not a purpose-built one.

Headline: at c1 decode is **101 tok/s** and at c4 it is **66 tok/s per request,
263-267 tok/s aggregate**. But this workload is prefill-bound — 82 of the 115 s
at c4 is prefill — so end-to-end output is only 50 tok/s at c1 and 71 tok/s at
c4. Both numbers are real; quote the one that matches the question.

## What was run

| | |
| --- | --- |
| Server | job `1238882`, the production c4 AutoRound W4G64 + MTP3 host, unmodified |
| Prompts | 16 unique, each exactly 16,383-16,384 tokens |
| Content | real PyTorch/vLLM source as context + a distinct code-generation task |
| Output | 512 tokens, forced with `--ignore-eos` |
| Sampling | greedy, `temperature 0` |
| Arms | `--max-concurrency 1` and `4`, 16 requests each |
| Repetitions | one excluded warmup, then 2 measured per arm |
| Client | inside the server's allocation via `srun --overlap`, so no network hop |

Prefix cache is reset before every measured repetition, and no two prompts
share a prefix, so no prefill is shortened. The server reported a **0.0%
prefix-cache hit rate** throughout, confirming it.

Correctness gate: the project's deterministic smoke prompt returned exactly
` Paris. Distance from Paris to Lyon is` (`semantic.json`).

## Results

Means over two repetitions.

| | c=1 | c=4 |
| --- | ---: | ---: |
| **Prefill** | | |
| TTFT mean | 5.153 s | 8.101 s |
| Prefill rate per request | 3,182 tok/s | 2,460 tok/s |
| **Decode, steady state** | | |
| Target step time | 28.11 ms | 42.9 ms |
| Tokens per target step | 2.849 | 2.843 |
| Decode per request | **101.3 tok/s** | **66.3 tok/s** |
| Decode aggregate | **101.3 tok/s** | **265.2 tok/s** |
| **End to end, this workload** | | |
| Mean TPOT | 9.889 ms | 40.011 ms |
| Output throughput | 50.16 tok/s | 70.98 tok/s |
| Total (in+out) throughput | 1,657 tok/s | 2,344 tok/s |
| Wall clock, 16 requests | 163.3 s | 115.4 s |
| **MTP3** | | |
| Draft acceptance | 61.63% | 61.51% |
| Per-position (r1) | 81.5 / 60.9 / 43.5% | 81.0 / 60.0 / 42.6% |

Reproducibility across the two repetitions: c1 is within 0.6% on every metric;
c4 is within 2.0% on decode and 5.1% on TTFT.

## Why the two decode numbers differ so much at c4

The client's inter-token latency is really inter-*chunk*: MTP3 delivers one
target step's accepted tokens at once, so 8,192 tokens arrive in 2,869 chunks
(2.855 per chunk, matching the 2.86 acceptance length the server reports).

At c4 those chunk gaps are bimodal:

| | count | mean |
| --- | ---: | ---: |
| Normal decode steps | 2,751 (95.9%) | 42.7 ms |
| Stalled steps | 118 (4.1%) | 1,805 ms |

The 118 stalls hold 213 s of the 330 s of summed gap time. They are **chunked
prefill of newly admitted requests**: `max_num_batched_tokens` is 8,192, a
16,384-token prompt is two chunks, and one chunk at the measured ~3,200 tok/s
prefill rate is ~2.5 s of the GPU doing no decode. No request was preempted
(`Preempt` never appears in the server log) and KV usage peaked at 4.1%, so
this is scheduling, not capacity.

The mean TPOT of 40 ms is therefore an average over a machine that spends most
of its time prefilling, not a decode rate. The 42.7 ms steady-state step is the
decode rate.

## The wall clock reconciles

Independent check that both readings are consistent, for c4:

| | |
| --- | ---: |
| Decode: 4 waves x 179 steps x 42.9 ms | 30.7 s |
| Prefill: 262,332 tokens at ~3,200 tok/s | 82.0 s |
| Predicted total | 112.7 s |
| **Measured** | **115.4 s** |

2.4% apart, which is the request ramp at the start and drain at the end.

## What this says about the workload

- **Prefill dominates.** 16K in and 512 out is 32:1, and prefill runs at
  ~3,200 tok/s against decode's effective ~265 tok/s aggregate, so 71% of the
  c4 wall clock is prefill. Concurrency 4 buys 1.44x end-to-end
  (50.2 -> 71.0 tok/s), far less than the 2.6x it buys on decode alone,
  because the prefills serialize behind each other.
- **Prefill does not overlap decode.** Every admitted request's two prefill
  chunks stop all four decode streams for ~1.8 s each. Raising
  `max_num_batched_tokens` would make the stalls fewer and longer; lowering it
  would make them more frequent and shorter, and neither changes the total. The
  fix, if this workload matters, is to let a prefill chunk share a step with
  decode tokens rather than displace them.
- **Acceptance is unaffected by concurrency**: 61.6% at c1 versus 61.5% at c4,
  and per-position 81/61/44 versus 81/60/43. Code generation accepts slightly
  worse than the 67.6% weighted average across mixed domains and slightly
  better than exact-400K synthetic runs.

## Files

| File | What |
| --- | --- |
| `make_prompts.py` | Builds the 16 exact-16,384-token prompts |
| `prompts.jsonl` | The prompt set |
| `run-benchmark.sh` | Runs both arms inside the server's allocation |
| `analyze.py` | Produces `summary.json` and the table above |
| `c{1,4}-r{1,2}.json` | Raw `vllm bench serve` results, detailed |
| `metrics-*.txt` | Server-side counters bracketing each repetition |
| `semantic.json` | Correctness gate output |
| `run.log` | Full run transcript |

## Reproduce

```bash
srun --jobid=<server-jobid> --overlap --ntasks=1 --cpu-bind=none \
  bash agent_space/experiments/2026-08-05-pytorch-16k-c1-c4/run-benchmark.sh
.venv/bin/python agent_space/experiments/2026-08-05-pytorch-16k-c1-c4/analyze.py
```
