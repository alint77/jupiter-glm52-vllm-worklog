# Scheduler: interleave prefill and decode steps (no mixed steps)

2026-10-11. vLLM `dc78b9b294` adds `VLLM_SCHED_INTERLEAVE_PREFILL=N`. With
N > 0, each step is one of two kinds:
- decode-only, or
- a single request's prefill chunk.

While requests are decoding, N decode-only steps run between prefill chunks.
N = 0 (the default) keeps the stock mixed steps.

Why: `../2026-10-11-prefill-nsys` measured a mixed 4K step at ~895 ms and a
pure 4K chunk at ~565 ms (c4). In a mixed step the prefill tokens go through
the DCP sparse decode kernel, which reads skip-layer KV from Grace. Only
single-request steps take the sparse prefill path.

## Setup

- c4 launcher config (`arm.sh`): MTP3 tp_sliced, 4 seqs, pool 2, reserve
  3.6, prefix caching. It runs from the frozen worktree `interleave`.
- `ab_launch.sh` runs every arm on one fresh hold:
  - node 2273430: order 0 1 2;
  - node 2273431: order 2 1 0.
- `bench_interleave.py` runs three windows, twice each on fresh text:
  - **mixed**: 3 decoders at ~10K context stream 1500 tokens. After 4 s, a
    68K-token prefill arrives (60K cached prefix plus 8K new).
  - **swarm**: 4 fresh 24K prompts arrive 1 s apart, 300 tokens each.
  - **fanout**: 4 prompts at once, sharing a cached 20K prefix plus 4K new
    each, 300 tokens each.
- GSM8K: 200 questions per arm. The policy changes which steps requests skip
  under async spec decode, so the math path is touched.

## Results

`ab_table.py`: the mean over 2 nodes × 2 reps.

| metric | k=0 (stock) | k=1 | k=2 |
| --- | ---: | ---: | ---: |
| mixed prefill TTFT s | 2.22 | **1.63** | 1.66 |
| mixed decoder max gap s | 0.93 | **0.65** | 0.65 |
| mixed decoder tok/s | 109.3 | **113.9** | 113.6 |
| swarm makespan s | 20.98 | **16.44** | 16.65 |
| swarm mean TTFT s | 9.78 | **7.28** | 7.44 |
| swarm max TTFT s | 15.58 | **11.09** | 11.43 |
| swarm decode tok/s | 52.8 | 59.0 | **61.1** |
| fanout makespan s | 6.15 | **5.47** | 5.52 |
| fanout mean TTFT s | 2.63 | **1.93** | 1.98 |
| fanout max TTFT s | 3.74 | **3.06** | 3.18 |
| fanout decode tok/s | **96.3** | 93.2 | 93.9 |
| GSM8K (node 30 / 31) | 0.910 / 0.920 | 0.900 / 0.905 | 0.905 / 0.910 |
| MTP acceptance length | 3.34 / 3.37 | 3.30 / 3.42 | 3.27 / 3.39 |

- Interleaving wins every window on TTFT and makespan:
  - prefill TTFT is -27% with decoders present;
  - swarm makespan is -22%;
  - fanout mean TTFT is -27%;
  - the longest decoder stall during a prefill drops from 0.93 s to 0.65 s.
- The only cost is fanout per-request decode rate (-3%). There the four
  requests decode while the others still prefill, and they now wait during
  pure prefill steps instead of riding along.
- k=1 and k=2 are within noise of each other. k=1 is marginally better on
  TTFT, and k=2 is marginally better on swarm decode rate.
- GSM8K and acceptance are within noise of stock.

## Recommendation

Enable `VLLM_SCHED_INTERLEAVE_PREFILL=1` in the sliced launchers. Not done
yet; it is the user's call.
