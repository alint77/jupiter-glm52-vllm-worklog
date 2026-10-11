# Prefill by step size: nsys breakdown vs roofline (4K budget)

2026-10-11. c4 config (MTP3 tp_sliced, 4 seqs, pool 2, `max_num_batched_tokens`
4096), worktree `interleave` (vLLM `dc78b9b294`) with
`VLLM_SCHED_INTERLEAVE_PREFILL=1`, so every window is pure single-request
prefill. `prof_sizes.py` runs one request per window (max_tokens 1). Run dir:
`/e/fscratch/profound/$USER/sliced-prof/pfs-b4096-2273589`; table from
`sizes_table.py` (mean over the 4 GPUs, ms).

The @60k prompts reuse a cached 60K prefix, so the new-token counts are 911,
1422 and 2394 tokens, and "4000" is two steps (4096 + 346).

Notes:
- The 8K-budget node was cancelled: an 8K budget needs HBM that would
  otherwise hold hot experts.
- nsys only wrote its reports after a hung TP1 worker was killed by hand
  (shutdown spin); the arm's trap should `kill -9` leftover workers.

| group | fresh 532 | @60k +911 | fresh 1013 | @60k +1422 | fresh 1997 | @60k +2394 | fresh 4053 | @60k +4096+346 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TTFT (client)** | 120 | **304** | 169 | **373** | 245 | **485** | 562 | **840** |
| GPU span | 117 | 199 | 165 | 262 | 241 | 380 | 559 | 727 |
| MoE gemm | 91.8 | 100.5 | 102.6 | 100.0 | 125.0 | 133.0 | 197.1 | 321.7 |
| MoE route/gather/act/combine | 6.4 | 10.2 | 10.3 | 13.3 | 19.2 | 23.1 | 39.4 | 43.7 |
| DCP q all-gather (NCCL) | 2.9 | 19.2 | 7.2 | 33.2 | 9.1 | 47.2 | 77.3 | 87.0 |
| sparse attention | - | 13.3 | - | 21.0 | - | 35.0 | 48.4 | 66.3 |
| DCP lse_combine | - | 10.9 | - | 16.8 | - | 28.1 | 47.3 | 52.1 |
| dense GEMMs (nvjet) | 14.6 | 19.5 | 20.1 | 26.7 | 39.8 | 44.1 | 75.9 | 86.8 |
| TP all-reduce (NCCL; one-shot below 1K is in "other") | 0 | 16.6 | 20.3 | 28.0 | 31.2 | 29.2 | 45.8 | 44.3 |
| indexer logits + top-k | 0.3 | 6.1 | 0.3 | 9.6 | 0.5 | 17.5 | 1.0 | 28.1 |
| triton glue + elementwise | 5.8 | 8.2 | 8.8 | 11.9 | 17.6 | 20.3 | 32.7 | 37.7 |

Fresh prompts of at most 2K tokens take the non-sparse attention path (no DCP
gather or combine).

## Findings

1. **~105 ms of host time per request at 60K context.**
   - TTFT minus GPU span is 3 ms for fresh prompts but 105-113 ms at every
     @60k size. It is not GPU work: the workers issue the step's first CUDA
     call as it starts, so the time is spent before the engine hands them
     the step (API server: chat template and tokenization of the whole
     ~180 KB prompt; engine core: block hashing, prefix lookup, scheduling).
   - For agentic turns (big cached prefix, small delta) this is 25-35% of
     TTFT, and it grows with context (~1.8 us per prompt token).
   - Not yet located: needs a probe (time /tokenize of the prompt; a fully
     cached request).
2. **MoE has a ~80 ms fixed cost per step plus ~28 us per token, and it is
   the GEMM kernel, not Grace.**
   - 532 tokens: 92 ms; 4053 tokens: 197 ms (plus 39 ms of route, gather,
     act and combine).
   - Cold prefetch (`VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=512`, 2 slots)
     is on and hidden. Each step copies 137 x 158 MiB from Grace on a side
     stream at 368-388 GB/s (~60 ms), with 0.1-3.5 ms of copy time when no
     kernel runs.
   - The GEMM reads every expert from HBM: 70.7 GiB hot plus the staged cold
     slot, ~98 GiB per step. At 532 tokens that is ~1.1 TB/s against
     ~3.4 TB/s achievable, a ~31 ms floor. At 4K it is 197 ms against a
     ~73 ms compute floor (37%).
   - (An earlier version of this note blamed serial Grace reads. The trace's
     memcpy table disproves that.)
3. **DCP attention communication grows with tokens and context.**
   - q all-gather plus lse_combine is 30 ms at +911 and 125 ms at a 4K step.
   - Floor at ~450 GB/s NVLink: ~75 ms at 4K, about 40% of the measured
     time. At 911 tokens the all-gather runs at only ~110 GB/s (message
     size).
4. **Dense GEMMs** are 76 ms at 4K against a ~48 ms floor. The TP all-reduce
   is 46 ms against ~26.
5. **Indexer** (60K context): 6-17 ms per step. Small.

Per 4K step: measured 559 ms against a ~270 ms floor on today's dataflow.
That floor is MoE ~85, DCP comms ~75, attention ~27, dense ~48, all-reduce
~26 and glue ~10.

## Levers, ranked

1. **MoE prefill GEMM efficiency.** The weights are already in HBM when the
   GEMM runs.
   - Small steps run memory-bound at ~1/3 of HBM bandwidth (92 ms against
     ~31). 4K steps run at 37% of bf16 compute (197 ms against ~73).
   - Up to -60 ms per small step and -120 ms per 4K step.
   - Next: ncu of the isolated kernel at 512 and 4096 tokens per tile class
     (achieved DRAM bandwidth, tensor pipe utilization, occupancy and
     stalls).
2. **Host time per request** (~105 ms at 60K, more at 100K+). Find it
   (tokenizer vs engine core) and cut it: incremental tokenization is not
   possible through the chat API, but hashing and lookup costs can be
   measured.
3. **DCP prefill dataflow.** A sequence-split all-to-all plus KV all-gather
   instead of the q all-gather plus LSE combine. About -70 ms at 4K and
   -15 ms at 1K.
4. Dense GEMMs and all-reduce: about -45 ms at 4K together.
