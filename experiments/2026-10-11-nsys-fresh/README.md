# Fresh Nsight Systems pass on prod: new prefill kernel + interleaved scheduler

2026-10-11. Prod as of vLLM `c8d61506be` (sliced prefill MoE v9) with
`VLLM_SCHED_INTERLEAVE_PREFILL=1` (serve.sh default since agent_space 73308e9),
frozen worktree `worktrees/nsys2`. Seven fresh holds in parallel
(`launch.sh`):

- decode, c = 1 2 4 8 16: `../2026-10-11-nsys-sol/nsys_arm.sh` (same launcher
  configs and ~5K / ~50K windows as that README), runs `nfr-c<c>-<job>`;
- prefill sizes at c4: `arm.sh <tag> sizes` (`../2026-10-11-prefill-sizes/
  prof_sizes.py`, 512-4000 tokens fresh and after a cached 60K), `nfr-sizes-2277153`;
- mixed / fanout at c4: `arm.sh <tag> mixed` (`prof_mixed.py`: the interleave
  A/B's mixed and fanout windows, each once un-profiled and once profiled),
  `nfr-mixed-2277154`.

Reports under `/e/fscratch/profound/$USER/sliced-prof/nfr-*`. Tables in `an/`
(`analyze.sh`: nsys-sol's `bd-*`, `kc-*`, `st-*`, `sol-w*.md`, `bd-w*.md`;
`sizes-table.md`). New tools: `moe_span.py` (per-layer prefill MoE wall span,
not summed PDL kernel time), `moe_copy.py` (span vs cold-prefetch copy
overlap), `step_groups.py` (kernel groups per prefill step).

## Decode: unchanged, the nsys-sol ranking stands

Aggregate tok/s per window is within ~2% of the nsys-sol run at every c (decode
kernels did not change). Step breakdowns (`an/bd-w1.md`) match nsys-sol's at
c1-c8 to ~0.1-0.4 ms. Two differences are artifacts:
- c1 ~50K window: MoE 7.84 vs 6.53 ms/step with identical residency
  (14774 hot); the ~5K window matches (6.40 vs 6.44). The generated text
  differs (acceptance 5.35 vs 3.65 tokens): content, not a regression.
- c16 (and 1-2 steps at c8): the windows contain prefill steps; see finding 2.

The decode levers in `../2026-10-11-nsys-sol/README.md` (dense GEMMs to HBM
floor, MoE at small M, launch-bound layer tail, C2C staging contention,
drafter path) are unchanged.

## Prefill by size (c4, 4K budget; `an/sizes-table.md`)

TTFT (client, ms, un-profiled mean of 2) and GPU span, vs the previous kernel
(`../2026-10-11-prefill-sizes`, same harness):

| prompt | TTFT before | TTFT now | GPU span before | now |
|---|--:|--:|--:|--:|
| fresh 532 | 120 | 127 | 117 | 116 |
| fresh 1013 | 169 | 209* | 165 | 159 |
| fresh 1997 | 245 | 248 | 241 | 229 |
| fresh 4053 | 562 | 513 | 559 | 516 |
| @60K +911 | 304 | 319 | 199 | 196 |
| @60K +2394 | 485 | 481 | 380 | 360 |
| @60K +4096+346 | 840 | 812 | 727 | 692 |

(*one of two runs 234 ms; GPU span is the cleaner number.) The new kernel
shows at 2K-4K (-12 to -43 ms per step); at <= 1K steps nothing moved.

Per-layer MoE wall span, served vs the isolated bench (same kernel):

| tokens | isolated (us) | served (us) |
|---|--:|--:|
| 512 | 604 | 866 |
| 1K | 807 | 997 |
| 2K | 1181 | 1222 |
| 4K | 1974 | 2048 |

## SOL efficiency (`an/sol-w2.md`, ~50K window; `an/sol-w1.md` ~5K)

Decode, us/call (min over ranks) and % of roofline. Rooflines and byte/FLOP
models are as in `../2026-10-11-nsys-sol`: HBM 3.6 TB/s, C2C 0.419 TB/s,
NVLink ~0.43 TB/s, bf16 ~630 TFLOP/s. The gap column is ms/step over the
roofline.

| kernel group | c1 | c2 | c4 | c8 | gap c1 / c4 / c8 (ms/step) |
|---|--:|--:|--:|--:|--:|
| step period (ms) | 19.81 | 20.26 | 25.64 | 38.58 | |
| MoE layer_kernel | 104.7 us, 59% | 104.0, 59% | 159.4, 75% | 254.0, 87% | 3.26 / 3.03 / 2.47 |
| dense GEMMs (8.92 GB/step) | 4.27 ms, 58% | 4.28, 58% | 4.58, 54% | 5.37, 46% | 1.80 / 2.10 / 2.90 |
| AR + RMSNorm | 7.3 us, 9% | 7.5, 9% | 8.9, 15% | 12.3, 22% | 1.00 / 1.18 / 1.50 |
| FlashMLA sparse | 10.9 us, 12% | 11.0, 12% | 13.8, 20% | 18.5, 29% | 0.74 / 0.86 / 1.02 |
| DCP gather_cat | 7.5 us, 13% | 7.5, 14% | 9.4, 22% | 12.4, 33% | 0.51 / 0.57 / 0.65 |
| DCP lse_reduce_scatter | 6.8 us, 13% | 6.7, 13% | 8.0, 23% | 10.9, 33% | 0.46 / 0.48 / 0.57 |

- These match nsys-sol to within noise, except c1 MoE at ~50K (59% vs 70%,
  content; the ~5K window matches).
- c16 is omitted. Its windows hold prefill steps (finding 2), so verify steps
  carry fewer than 64 tokens. MoE then shows 115-124% "SOL" against the
  64-token floor, which is not meaningful. nsys-sol's c16 (98% MoE, dense 45%,
  AR 32%) stands.
- Small collectives and attention sit at 9-33%. They are latency-bound
  (2-18 us kernels), so the lever is fewer launches, not bandwidth.

Prefill (c4, `step_groups.py`, `moe_span.py`):

| item | measured | floor | % SOL | floor basis |
|---|--:|--:|--:|---|
| 4096-token step | 516 ms | ~270 ms | 52% | prefill-sizes README dataflow floor |
| MoE, 4K step (75 layers) | 153 ms (2048 us/layer) | 74 ms (982 us/layer) | 48% | bf16 compute |
| MoE, 512-token step | 63.5 ms | ~63 ms | ~100% | 24.3 GiB cold experts over C2C |
| 384/512-token step | 113-125 ms | ~63 ms C2C + ~50 ms non-MoE | | per-step fixed cost |
| DCP q all-gather + lse_combine, 4K | 125 ms | ~75 ms | 60% | NVLink ~450 GB/s |
| dense GEMMs, 4K | 78 ms | ~48 ms | 62% | bf16 compute |

## Findings

### 1. Every prefill step pays ~115 ms, set by C2C, and agentic turns add a step

- A prefill step of 384-512 tokens costs 113-125 ms of GPU at 20-68K context
  (`step_groups.py` on the fanout window), against ~510 ms for 4096 tokens:
  ~75 ms fixed plus ~0.107 ms per token.
- The fixed part is mostly MoE, and that MoE is at its C2C floor. Every step
  touches all 65 cold experts of every layer: 24.3 GiB per GPU per step over
  C2C at ~390 GB/s, ~63 ms.
  - At >= 512 tokens the 2-slot prefetch copies them (0.86 ms per layer).
  - Below 512 (`VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS`) the GEMM reads
    them in place.
  - Both take the same ~117 ms per step. The fresh-512 MoE wall total is
    63.5 ms per step, i.e. at that floor.
  - The served-vs-isolated gap at 512/1K (+260 / +190 us per layer) comes
    from layers that overlap the copies: 869 us overlapped vs 713 us
    copy-free (`moe_copy.py`; only 3-7 copy-free layers, a lead not a
    result). At 2K/4K the copies land outside the MoE and served ≈ isolated.
- **Every cached-prefix request recomputes 257-512 tokens.** Requests whose
  whole prompt is cached (c16 windows) run 384/512-token padded prefill
  steps. Two things combine:
  - the EAGLE/MTP last-block drop (`kv_cache_coordinator.py:98`);
  - DCP4 hashing in 64 x 4 = 256-token blocks.

  The result is the partial block plus one full 256-token block. Costs:
  - ~27 ms per turn in-step;
  - when the delta is close to 4K, a whole extra ~115 ms step. Fanout
    (20K cached + 4K new) runs 4096 + 384/512 per request. The mixed case
    (60K + 8K) runs 4096 + 4096 + 384.

### 2. Interleave serializes bursts of small prefills (regression at c >= 8)

- Under `VLLM_SCHED_INTERLEAVE_PREFILL=1` a prefill step holds one request,
  so N requests arriving together pay N fixed costs in a row, each followed
  by a decode step.
- In the c16 run (16 cached prompts arriving at once), prefill steps of
  384/512 tokens repeat every ~175 ms. They are still running 2.8 s after
  arrival: 9-10 per 2 s window.
- The old mixed policy (nsys-sol run) finished the same startup in ~1.35 s:
  its c16 windows hold 0 and 1 prefill steps.
- Steady-state decode rate is unaffected. TTFT for bursty arrivals at c >= 8
  is about 2x worse.
- Why single-request: under DCP the bf16 sparse prefill path needs
  `num_reqs == 1` (`flashmla_sparse.py:430`, `_dcp_prefill_local_len`).
  Multi-request steps take the fp8 mixed path, which costs ~80 us per token
  more at 4K.

### 3. Host time per request at long context

`probe_host.py` (`launch_host.sh`, hold 2278194, no profiler, idle server).
Each request is a cached prompt plus a new short suffix, max_tokens 1.
Values are ms, mean of reps 1-2:

| prompt tokens | client TTFT | server TTFT | prefill time | /tokenize (chat) | /tokenize (raw) |
|--:|--:|--:|--:|--:|--:|
| 1K | 119 | 118 | 116 | 2 | 2 |
| 10K | 135 | 128 | 115 | 14 | 13 |
| 30K | 176 | 159 | 121 | 41 | 40 |
| 60K | 234 | 201 | 126 | 82 | 80 |
| 100K | 303 | 252 | 136 | 128 | 124 |

(The two `/tokenize` columns: chat applies the chat template, raw is the
prompt string only.)

- The ~105 ms found in `../2026-10-11-prefill-sizes` is located:
  - **Tokenizing the whole prompt again every turn: ~1.3 us per token**,
    80 ms at 60K and 128 ms at 100K. Server TTFT minus prefill time matches
    `/tokenize` within ~5 ms. The chat template adds ~2-5 ms; the tokenizer
    is the cost.
  - **30-50 ms before the request reaches the engine** (client minus server
    TTFT: HTTP, a 180-300 KB JSON body, streaming setup). It grows with
    prompt size. Not yet split further.
- Queue time is 0. The prefill step is the fixed ~115-136 ms of finding 1
  at every size from 1K to 100K.
- An agentic turn at 60K context: 234 ms TTFT = 126 GPU + ~80 tokenizer
  + ~30 front end. At 100K, host work (~165 ms) exceeds GPU work (136).

## Levers, ranked for agentic traffic (cached prefix + small delta)

Gains are per turn or per burst. They are estimates from the traces and
probes above, not yet A/B'd.

| # | lever | gain | where | cost / risk |
|--:|---|---|---|---|
| 1 | **Tokenizer cache for repeated prompt text.** Tokenize per segment between special tokens (GLM's chat template puts role tokens between messages, and the tokenizer splits on special tokens), and cache segment hash -> ids. Agentic turns resend the same messages. | -70 to -120 ms TTFT per turn at 60-100K, every turn, at every c | API server (`chat_utils` / tokenizer wrapper) | Token ids must stay identical: verify on the captured agentic set. No GPU or numerics change. |
| 2 | **Multi-request prefill steps.** Fill a pure-prefill step with chunks of several requests up to the budget: still no decodes, but more than one request. Needs the DCP bf16 sparse prefill for N requests (workspace per request; `triton_filter_and_convert_dcp_index` already takes per-request workspace ids and starts). | c16 burst of cached prompts: ~2.8 s -> ~1 s to last TTFT. Fanout: the 4 x ~115 ms tail steps fold into the next request's chunk (~-0.3 s makespan, -0.1 to -0.4 s TTFT). | scheduler `_interleave_step_kind` + `flashmla_sparse.py` DCP prefill | Moderate. The interim policy "pack only when every chunk is small, fp8 mixed path" needs measuring first. |
| 3 | **Stop recomputing 257-512 cached tokens.** The EAGLE/MTP last-block drop runs at the 256-token DCP hash block. Drop at the 64-token attention block, or recompute only what the MTP layer needs. | ~-27 ms per turn. -115 ms (a whole step) when the delta is within ~500 tokens below a 4K multiple. | `kv_cache_coordinator.py` eagle drop, MTP state | Correctness of the drafter's KV on the dropped block: needs care plus an acceptance check. |
| 4 | **Front end before the engine** (30-50 ms at 60-100K). | -20 to -40 ms per turn | API server: py-spy one request first | Unknown until profiled. |
| 5 | **Small prefill steps vs the C2C floor.** ~63 ms of the ~115 ms is cold experts over C2C; the rest is attention / DCP comms / dense / glue (~50 ms). Overlap the copy with the non-MoE half of the layer, and check the +150 us per layer seen when the MoE overlaps the copy (`moe_copy.py`, few samples). | <= ~12 ms per 512-token step from contention; more from the overlap restructure | tiered prefetch scheduling | Needs a controlled probe (MoE bench with and without a concurrent cold copy). |
| 6 | Decode levers of `../2026-10-11-nsys-sol` (dense GEMMs to floor, MoE at small M, launch-bound layer tail, C2C staging contention, drafter path) | as there | | unchanged |

For the agentic workload (cached prefix plus a small delta, bursts of
subagents), 1-3 together remove roughly 100-250 ms from a 60K-context turn's
~234 ms + steps. Lever 2 also removes the c >= 8 burst regression that
interleave introduced.
