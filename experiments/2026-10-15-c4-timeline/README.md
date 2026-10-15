# c=4 agentic serving, end to end: where every millisecond goes

2026-10-15. Prod c4 config (sliced MTP3, 4 seqs, KV pool 2, interleave 1,
prefetch 384, fastokens; vLLM c8d61506be frozen worktree nsys2) on hold
2333343. `timeline_run.py`: 4 closed-loop agents replay 48 consecutive Claude
Code turns each (cc-replay.jsonl, session 898193f4, offsets 0/48/96/144;
first contexts 58K/156K/237K/294K), streamed on /v1/messages, max_tokens 2048,
zero think time. Sampler: scheduler gauges and node every 0.5 s. nsys windows
at 60 s and 170 s (6 s each). py-spy (non-blocking, active samples only) on
the API server and EngineCore for the run, worker 0 for 70 s around window 2.
Run 1 died at 60 s (`/start_profile` needs POST); run 2 is the data
(`sliced-prof/c4tl-2333343`, run 1 in `run1/`).

Tools: `reqs.py` (requests + 10 s bins), `step_timeline.py` (per engine step:
kind, wall, GPU busy/idle, idle gaps with the worker's CUDA API calls, kernel
groups, memcpys), `pyspy_agg.py` (py-spy per thread, leaf/inclusive, spin and
idle split), `prefix_diverge.py` (renders consecutive turns through vLLM's
Anthropic conversion + chat template, finds the first differing token).

## Request path, measured cost per request

| stage | where | cost |
|---|---|---|
| HTTP + JSON parse of a 0.2-1.2 MB body, Anthropic -> chat conversion | API server main thread (uvloop) | inside ~5 s active CPU for the whole 270 s run (~2% of a core) |
| chat template (jinja) + tokenize (fastokens) | API server thread pool | ~12 ms/request (render ~6, tokenize ~2) |
| block hashing for prefix cache (sha256 per block) | EngineCore input thread | ~10 ms/request at 150-300K tokens |
| schedule() + update_from_output | EngineCore main | ~1% of its samples; the rest spins on the workers' shm queue (sched_yield) |
| detokenize + stream SSE | API server main | ~40% of the API server's ~5 s |
| GPU steps | workers | everything else |

The CPU pipeline is not a bottleneck. Every worker and EngineCore spins a
core (sched_yield on the shm broadcast queue); harmless on 288 cores.

## Timeline of the run (`reqs.py`)

- 0-110 s: four cold sessions (58K, 156K, 237K, 294K tokens) prefill one at a
  time (FCFS, interleave admits one prefilling request). TTFT 8 / 26 / 56 /
  110 s. Meanwhile generation runs at 1.5-14 tok/s total.
- 110-270 s: follow-up turns. Turns with a cached prefix: TTFT 0.34-1.2 s,
  decode 93-144 tok/s per request. Turns that lost their prefix: TTFT 33-69 s.
- KV pool: 800,255 tokens (pool seqs 2 x 400K) vs 4 live contexts of
  70-300K: usage 82-87% whenever all four are resident.

## Engine steps (`step_timeline.py`, device 0)

Window 2 (t = 170-176 s; one ~240K request prefilling, three decoding):
104 steps in 6.04 s.

- **Prefill chunk** (4096 tokens at ~240K context): 674 ms wall (735-763 ms
  steady), GPU busy 596 ms, idle 78 ms (12%).
  - kernel time (ms, summed over streams): MoE 221, indexer 96, NCCL
    all-gather 93, dense GEMM 66, NCCL all-reduce 57, sparse attention 52,
    norms/elementwise 28, DCP combine 9, KV gather 6, drafter 3.
  - H2D 24.3 GiB/step (cold-expert prefetch, hidden).
  - step prep: ~4.7 ms of `_build_prefill_chunk_metadata` + DeviceScan with
    16 tiny D2H copies (8 host round trips) before the forward.
- **Decode step**: verify 18.0-18.6 ms during prefill (fewer tokens), 22.4-24.4
  ms with 4 decoders at 160-300K. GPU idle 0.3-0.4 ms (GPU-bound). Plus 4 small
  drafter/prep steps of ~0.6 ms.
  - verify graph per step (us): MoE layer_kernel 10,581; decode_gemm 3,540;
    AR+RMS 1,690; skip-layer KV rows from Grace (`_gather_rows` x19 + remap/
    compact/mark) 1,690; FlashMLA 1,098 + combine 521; finalize 1,059; DCP
    gather_cat 755 + lse_rs 632; route_prep 430; indexer ~1,300 (side stream).
- **Interleave cadence**: prefill chunk, then exactly one decode step (and
  its drafter passes), repeat. Decode gets ~19 ms per ~770 ms: 2.5% duty,
  ~4 tok/s per decoder while any long prefill runs.

Window 1 (t = 60-67 s): prefill 574 ms/step (11 steps), decode 18.2 ms.

## Inefficiencies, ranked by wall time lost in this run

1. **Triton recompiles a kernel on every prefill step.**
   `_pack_dcp_topk_candidates_triton_kernel`
   (`vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py:70`)
   declares all strides `tl.constexpr`; `logits_stride0` is the context
   length, new every chunk. Each prefill step: GPU idle 70-94 ms (full
   compile; py-spy: make_ttgir -> make_llir -> make_ptx -> ptxas under
   `pack_dcp_topk_candidates_cutedsl`) or 13-16 ms (disk-cache hit, window 1
   re-serving run 1's shapes). The Triton cache holds **6,401 variants** of
   this kernel. Cost: ~12% of every long-context prefill step, and every
   agentic turn's TTFT pays it once. Fix: strides as plain ints (Triton
   specializes divisibility only); same math.
2. **Prefix cache loses append-only prefixes.** Server hits vs the rendered
   prompts (`prefix_diverge.py`, token counts match the server's to ~30):
   - #63 -> #64 (156K): 23% hit, waited 50 s behind a 294K cold prefill, then
     recomputed ~120K: TTFT 69 s. Consistent with LRU eviction of the freed
     prefix while other requests grew (800K-token pool, ~750K live).
   - #113 -> #114 (241K): rendered 100% append-only, server hit 15% (37K):
     ~204K recomputed, TTFT 33.6 s. KV was 43% used at that moment, so pool
     pressure alone does not explain it. **Root cause (found later):** the
     Anthropic frontend merged Claude Code's inline system reminders into
     the leading system block (no `--chat-template` -> merge), shifting the
     prompt after ~35K; fixed by upstream #58754 (fork 144f2f4aee), see
     `../2026-10-15-prefix-miss/`.
   - #16 -> #17 (76K): append-only, hit 46%.
   - #160 -> #161 (302K): real client-side change: Claude Code's tool list
     gained `mcp__ide__executeCode` mid-list; tools render before the
     conversation, so 8% hit, TTFT 47 s.
3. **Head-of-line blocking of prefills.** One prefilling request at a time,
   FCFS: a +565-token follow-up turn (#64) waited ~50 s behind a 294K cold
   prefill. Short-suffix turns should be able to go before a long cold one
   (chunk-level round robin or shortest-remaining-first).
4. **Decode starvation during long prefills.** Interleave 1 gives decoders
   2.5% of the GPU while a 240K prefill runs (~4 tok/s each, for up to 50
   s). The knob is per prefill chunk; at long context a chunk is ~0.75 s.
5. **Long-context prefill step composition.** At ~240K context vs ~60K: the
   indexer (mqa logits over all keys for 4K queries) grows 32 -> 96 ms per
   chunk; NCCL all-gather 82-93 ms + all-reduce 50-57 ms per chunk (DCP q
   gather + TP reduce) are ~22% of the step.
6. **Skip-layer KV staging from Grace in decode**: ~1.7 ms/step (7.5%) at
   160-300K contexts (19 gathers), not "free" at these lengths.
7. **Prefill step prep**: ~4.7 ms of metadata kernels with 8 host round trips
   per prefill step (<1% at 4K chunks; matters for short turns).

Not inefficient here: API server, tokenizer/template, scheduler CPU, IPC,
cold-expert prefetch (H2D fully hidden), decode launch overhead (verify graph,
0.3-0.4 ms idle/step).

## Fixes and follow-ups (same day)

Holds 2334599/2334600/2334606 (fix arms), 2334590/2334591 (SRPT A/B).

- **Triton recompile: fixed** (vLLM 7284232eb5, strides as runtime ints).
  Served check: a fresh Triton cache holds 24 entries for the candidate-pack
  kernel after a 600 s c=4 timeline on the fix (fixed set, no growth), vs 232
  after only the short TTFT sweep on the old code. `ttft_sizes.py` cannot show
  the TTFT win (its shapes repeat; rep 4 old vs new equal within noise:
  fresh 512-4000 114/240/246/498 vs 119/144/248/503 ms rep 1); the per-step
  cost was measured in the nsys windows above (13-94 ms per long-context
  prefill chunk).
- **Prefix misses reproduce at c=1 on an empty pool** (`prefix_repro.py`,
  `logs/fix-pfx-2334600.log`): #112 cold, #113 hits 99.8% of it, but #114
  (append-only) hits 37,376 of 241K; #16 and #17 both hit 35,072; #63/#64
  36,096/36,352; #160/#161 24,320/22,784. Not eviction, not concurrency.
  **Root cause: the server's own prompt.** Without `--chat-template`,
  `/v1/messages` merged Claude Code's inline `role: system` reminders into
  the leading system block, which GLM renders right after the tools (~35K);
  each new reminder shifts the rest of the conversation. (The earlier
  "rendered prompts are append-only" check went through paths that do not
  merge.) Fix: cherry-pick of upstream #58754, fork 144f2f4aee. Served, 600 s
  c=4 timeline: 69 -> 146 turns, 97 -> 229 out tok/s, TTFT mean 9.10 -> 2.85
  s, p90 33.8 -> 1.47 s (`../2026-10-15-prefix-miss/`).
- **SRPT prefill order: no win** (vLLM 877df9b262, `VLLM_SCHED_INTERLEAVE_SRPT`,
  default off). 600 s timeline, `ab_srpt.py`:

  | arm | turns | out tok/s | prefix hit | TTFT mean / p50 / p90 (s) | follow-up mean |
  |---|---|---|---|---|---|
  | FCFS | 69 | 97.3 | 75.9% | 9.10 / 0.58 / 33.8 | 6.80 |
  | SRPT | 64 | 89.7 | 74.9% | 9.29 / 0.69 / 39.0 | 6.38 |

  The slow turns are the same in both arms: deterministic prefix misses that
  recompute 150-300K tokens. Waiting behind another prefill is second order
  next to them. On the prefix fix SRPT wins on follow-ups (mean 1.07 vs
  1.92 s, 155 vs 146 turns; `../2026-10-15-prefix-miss/`).
- **Prefill prep** (7 indexer metadata builds per step, ~0.68 ms each under
  nsys): CPU dispatch of ~25 small torch ops per build, not the two host
  syncs (6 us each). ~1% of a short turn; deferred.
