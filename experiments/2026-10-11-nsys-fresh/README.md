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

### fastokens (`VLLM_USE_FASTOKENS=1`): measured

- **Tokenizer bench** (`tok_bench.py`, `tok_bench2.py`, Booster node, no
  server). GLM-5.3 already loads the Rust HF `tokenizers` 0.22.2, at ~1.4-1.6
  us per token. fastokens 0.3.3 runs in ~1 ms for distinct 64K-token texts
  and 0.7 ms for a 40K-prefix turn with a new suffix (HF: ~100 / 63 ms). Its
  first call is ~28 ms (warm-up; it caches across calls).
- **Identity** (`tok_ident.py`): 0 mismatches over 17.7M tokens. That covers
  all 5,643 text files in the repo, unicode/emoji/control stress strings,
  and 20 chat-templated turns with system/tool messages and tool calls.
  Decode matches too.
- **Served A/B**, same time on two fresh holds (`launch_host.sh`;
  `nfr-host-hf-2280330` vs `nfr-host-ft-2280329`). Fastokens installed into
  `/e/fscratch/profound/$USER/pylibs/fastokens` (not the venv), on
  PYTHONPATH. Client TTFT in ms, mean of reps 1-2; the change column is the
  fastokens delta:

| prompt | HF tokenizers | fastokens | change | server /tokenize HF -> ft |
|--:|--:|--:|--:|--:|
| 1K | 123 | 120 | -3 | 2 -> 1 |
| 10K | 136 | 124 | -12 | 14 -> 3 |
| 30K | 175 | 137 | -38 | 41 -> 6 |
| 60K | 234 | 154 | **-80** | 84 -> 11 |
| 100K | 299 | 181 | **-118** | 127 -> 17 |

- Prefill time (GPU step) is unchanged. The remaining host time at 60K is
  ~22 ms before the engine (client minus server TTFT), plus ~6 ms
  between engine arrival and scheduling.
- **Shipped.** fastokens 0.3.3 is installed in the venv and serve.sh
  exports `VLLM_USE_FASTOKENS=1`.
  - Prod-path check (`nfr-host-prod-2281365`, no scratch PYTHONPATH):
    60K 152 ms, 100K 175 ms client TTFT, matching the A/B.
  - The server log shows "[fastokens] patch_transformers: successfully
    patched".


## Prefill breakdown and floors (`prefill_bd.py`)

Wall time per prefill step: each instant goes to the highest-priority
category with a kernel running, so PDL overlap and side streams are not
double-counted. Values are ms, mean of 4 GPUs, c4 config. The windows
come from `nfr-sizes-2277153`:

- the agentic turn: a cached 60K prefix plus 911 new tokens (incl. the
  257-512 recomputed), padded to a 1024-token step (window 2);
- a 384-token tail step after a 4K chunk at 60K (window 8, step 1);
- a full 4096-token chunk at 60K (window 8, step 0).

| category | turn @60K, 1024 | tail @60K, 384 | chunk @60K, 4096 | floor (1024 / 4096) | floor basis |
|---|--:|--:|--:|--:|---|
| **step wall** | **195** | **134** | **536** | | |
| MoE GEMM | 64.5 | 69.2 | 127.8 | 30 / 74 | max(weights 102 GB per step / HBM, 18 us/token bf16). The 384 step has no prefetch: C2C 63 |
| MoE route/gather/act/combine | 7.1 | 3.1 | 24.7 | ~fusion only | gather and combine are at HBM speed (~118 us each per layer at 4K) |
| DCP q all-gather (NCCL) | 19.4 | 9.7 | 76.9 | 11 / 38 | q T x 64 h x 576 x 2 B, 3/4 inbound, 450 GB/s |
| DCP lse_combine | 12.4 | 5.7 | 51.0 | 8 / 33 | outputs T x 64 h x 512 x 2 B reduce-scatter |
| dense GEMM | 23.2 | 12.6 | 89.3 | 12 / 48 | bf16 compute |
| TP all-reduce | 18.5 | 7.4 | 45.6 | 6 / 25 | 2 x T x 6144 x 2 B per layer, NVLink |
| attention (sparse fwd) | 13.7 | 5.7 | 60.4 | ~10 / ~36-45 | 2048 keys x 64 h per token, local quarter |
| indexer | 7.7 | 4.8 | 20.5 | | |
| glue (triton norm/rope/quant, elementwise, KV write) | 8.5 | 4.7 | 31.4 | | |
| MTP drafter MoE (Marlin) | 1.3 | 4.3 | 0 | | |
| router top-k | 0.6 | 0.5 | 1.7 | | |
| idle (launch gaps) | 18.2 | 6.9 | 7.1 | ~0 | |

- Outside the GPU, per request at 60K: tokenizer ~80 ms and front end ~30 ms
  (finding 3). Client TTFT for the 1024 turn is 303 ms against 195 ms of GPU.
- Other sizes are in the same proportions (`prefill_bd.py` on windows 1-8):
  - fresh 640: 116 ms, MoE 57;
  - @60K 1422: 253 ms, q all-gather 39;
  - @60K 2394: 344 ms;
  - fresh 4053: 504 ms.
- Steps above 1024 tokens run eagerly (captures end at 1024). Idle is 7-14 ms
  there; at the captured 1024 step it is 18 ms (the eager attention part
  between piecewise graphs).

### Opportunities (ms saved per step, from the table; estimates, not A/B'd)

| # | lever | turn @60K 1024 | 4K chunk | notes |
|--:|---|--:|--:|---|
| P1 | **Tokenizer (fastokens, `VLLM_USE_FASTOKENS=1`)** | -80 (measured, host) | -80 to -118 per request at 60-100K | ids identical on 17.7M tokens; served A/B done (above) |
| P2 | **DCP prefill dataflow: all-gather the context KV instead of q + LSE combine.** Each rank then attends its own 16 heads over the full KV: 60K x 656 B = 42 MB per layer, ~7-10 ms per step at any T. | -22 | -118 | Biggest GPU lever at 2-4K. Same math, different reduction order (one pass vs 4 partials + LSE merge): not bitwise, needs an eval. Transient workspace ~42 MB per layer at 64K. |
| P3 | **MoE GEMM at small T toward the HBM floor** (runs at 1.6 TB/s at 1K; isolated 807 vs served 860 us per layer) plus the 4K compute gap (48%) | -34 | -54 | kernel work; copy contention (+150 us per layer, finding 1) is part of the small-T gap |
| P4 | **Fewer prefill steps**: multi-request prefill steps (finding 2), no 257-512-token recompute (finding 1) | -27 per turn; -134 when a turn crosses 4K | | scheduler + DCP sparse prefill for N requests; MTP last-block drop |
| P5 | **Dense GEMMs** (52-62% of bf16) | -11 | -35 | cuBLAS nvjet tiles; tune or a custom GEMM at these shapes |
| P6 | **TP all-reduce** (NCCL ring LL, ~55% of NVLink) | -12 | -21 | custom two-shot AR or AR+RMSNorm fusion at prefill sizes (decode already uses trtllm fusion) |
| P7 | **Launch gaps** (idle) | -13 | -5 | capture more of the step; the eager attention part |
| P8 | **Cold prefetch for the 384-token tails** (threshold 512 -> 256) | -20 per tail step | | env knob only: the tail's MoE then reads HBM instead of C2C (69 ms). Cheapest A/B. |
| P9 | glue + MoE gather/combine fusion, indexer, MTP drafter MoE on the tiered kernel | ~-10 | ~-30 | many small items |

For the agentic turn (303 ms TTFT at 60K, 1024 step), P1 + P2 + P3 + P6 +
P7 alone come to ~-160 ms (to ~145 ms). For a 4K chunk, P2 + P3 + P5 + P6 come
to ~-230 ms of 536 (to ~305 ms).

## Findings

### 1. Every prefill step pays ~115 ms, set by C2C, and agentic turns add a step

- A prefill step of 384-512 tokens costs 113-125 ms of GPU at 20-68K context
  (`step_groups.py` on the fanout window), against ~510 ms for 4096 tokens:
  ~75 ms fixed plus ~0.107 ms per token.
- Every step touches all 65 cold experts of every layer: 24.3 GiB per GPU
  over C2C at ~390 GB/s, ~63 ms. That bounds the whole step from below.
  - At >= 512 tokens a 2-slot prefetch copies the cold experts on a side
    stream (0.86 ms per layer), overlapped with the whole step.
  - Below 512 (`VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS`) the GEMM reads
    them in place over C2C, so it is C2C-bound itself. The 384-token tails
    show 69 ms of MoE GEMM.
  - (Corrected: an earlier version of this note called the MoE GEMM at 512
    "at its C2C floor". With prefetch it reads HBM. Its floor is
    max(102 GB / 3.4 TB/s = 30 ms, compute) per step, and it runs at ~50%
    of that at <= 1K tokens. See the prefill breakdown.)
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
