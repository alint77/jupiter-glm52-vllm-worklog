# GLM-5.3 W4A16, MTP K=7, c=1, 400K: fresh decode profile and the INT4 one-kernel port

Back on GLM after the MiMo work (`2026-09-27-dak-tiered-moe-kernel`). Every
GLM breakdown so far is MTP3 at 96K and predates the MiMo-era changes, so the
first step is a new profile of the shape we now want to serve:

- **MTP K=7, not DFlash2.** DFlash2 is overfit to GSM8K-like text (5.6 AL there
  against 3.10 on Claude-Code-shaped traffic, `2026-09-04-spec-comparison`),
  runs eager because its captured graph loses acceptance, and needs an
  unbudgeted draft KV cache. MTP7 measured 3.24 AL on the same traffic.
  It verifies 8 tokens per step, the same shape as MiMo's DFlash K=7.
- c=1, DCP1, 400K context, reserve 10, cold prefetch at 1024 tokens, prefix
  caching off so a repeated prompt really prefills.

## Files

- `serve.sh` -- the server (the spec-comparison mtp7 arm plus cold prefetch)
- `capture.py` -- decode profiles at short, 96K and ~290K context, each with an
  unprofiled control on the same request
- `run.sh` -- serve + capture on a held node (`hold.sbatch`, `onnode.sh`)
- analysis: `../2026-09-26-mimo-decode-profile/{analyze,imbalance,gaps}.py`,
  which now also separate out the DSA indexer

## Sanity check of the analyzer on the old MTP3 trace (job 1665068, 96K)

| target graph (ms/step) | |
|---|---|
| cold Marlin (Grace) | 9.94 |
| hot Marlin (HBM) | 1.97 |
| TP all-reduce | 7.09, of which **6.50 waiting** for the slowest GPU |
| dense GEMM | 3.71 |
| norm/rope/elementwise | 3.65 |
| attention (sparse MLA) | 1.84 |
| MoE routing/align/sum/act | 1.48 |
| DSA indexer | 0.46 |

## INT4 one-kernel decode MoE

The one-kernel decode path (`tiered_decode.cu`) was MXFP4-only. GLM-5.3 W4A16
has the same expert shape (6144 x 2048, top-8), so the port is the weight
format only:

- weights: the same Marlin 4-bit repack, so the same nibble order; uint4b8
  decodes exactly as `hfma2(0x6400 | code, 2^-14, -1032 * 2^-14)`, landing at
  the `value * 2^-14` the MXFP4 decode uses, so nothing downstream changes
- scales: bf16 in `marlin_permute_scales` order, where a lane's rows g and
  g+8 of block `mb` are one 32-bit word at element `8g + 2mb`; a stage carries
  4 KB of scales instead of 2 KB (215 KB of shared memory)
- the path now also runs with replicas off (static slot maps) and with a
  shared expert the runner executes itself (GLM's case)
- tests: `tests/kernels/moe/test_tiered_decode_moe.py` runs both formats;
  INT4 max error 2.9e-3 to 3.1e-3 of the row max against Marlin's 6.0e-3 to
  7.6e-3 (T = 1, 3, 8), MXFP4 unchanged. vLLM commit `12a52eacc7`.

## Results (jobs 2096362, 2096391; MTP7, c=1, 400K, reserve 10)

Unprofiled step time, 6 s windows on sampled (T=1) text. **These windows move
by +-3-5 ms with the generated content**, so only same-node rows compare, and
only coarsely; the greedy A/B below is the controlled measurement.

| node | arm | short | 96K | 288K |
|---|---|---|---|---|
| 2096391 | Marlin | 45.1 | 45.8 | 46.2 |
| 2096391 | INT4 one-kernel | 45.8 | 45.8 | 45.5 |
| 2096362 | Marlin | 46.2 | 40.6 | 41.1 |

Context length does not move the step: 288K costs what short does.

### Where a step goes (profiled, Marlin, short; profiler adds ~25%)

| bucket | ms/step |
|---|---|
| cold Marlin (Grace) | 17.5 (hot+cold union 20.5) |
| hot Marlin (HBM) | 2.6 |
| waiting for the slowest GPU after MoE | 8.0 -- **cold** spread 8.07, hot 1.63 |
| MTP7 drafting | 8.9, of which ~3.3 GPU work |
| dense GEMM | 3.9 |
| attention (sparse MLA) | 2.0 |
| MoE routing/align/sum/act | 1.5 |
| DSA indexer | 0.28 short, 0.58 at 96K, 1.28 at 288K |

1. **The decode MoE is bound by reading cold experts over C2C.** 17.5 ms of
   cold Marlin over 75 layers is ~233 us per layer, ~4 cold experts (20.3 MiB)
   per rank per layer at ~380 GB/s. MiMo reads 1-3.
2. **The waiting is cold imbalance**, rotating across all four ranks (22-28%
   last to arrive each). Replicas with balancing are the lever; the GLM
   replica campaign (`2026-09-05-decode-placement-replicas`) measured -9%.
3. **The INT4 one-kernel path does not change end to end.** With PDL off its
   MoE is 19.7 ms against Marlin's ~22.0 (w13 12.5, w2 6.8, small kernels
   0.5), but the time saved turns into waiting for the slowest rank. It is a
   prerequisite for in-kernel time balancing, not a win by itself.
4. **The DSA indexer does grow with context** (0.28 -> 1.28 ms), but is small.
5. ~~Prefill ran at 9,565 tok/s~~ -- wrong: that was the server's 10 s
   logging average. Timed directly (`bench.py`), a 96K prompt takes 19.7 s to
   first token, ~4.9K tok/s, against 4,145 tok/s (23.2 s) on 09-04.

### MTP drafted eagerly: a capture-size gap, not a choice

Only 2 CUDA graph launches per step, ~430 eager kernel launches, and a
drafting phase that is ~65% GPU idle. The autoregressive speculator captures
two routines: the draft prefill (position 0, K+1 tokens) and a separate graph
for the draft decodes (positions 1..K-1, **one token each**). The decode
manager only captures sizes from `cudagraph_capture_sizes` that fit
`max_num_reqs x 1` (`v1/worker/gpu/cudagraph_utils.py:229-236`). At c=1
with `cudagraph_capture_sizes: [8]` there are none, so the graph is skipped
without a word and six of the seven MTP passes run eager every step.

The c=1 MTP profiles before this one had the same gap (the 09-04 MTP3 profile
captured `[4]`); the c=4 production launcher escapes it because
`[4, 8, 12, 16]` contains 4 = max_num_seqs x 1.

Fix: `cudagraph_capture_sizes: [1, 8]` (`serve.sh`, `CAPTURE_SIZES`). Job
2096391, short context, profiled: graph launches 2 -> 8 per step, eager
launches ~430 -> ~63, drafting 8.3-10 -> 3.3 ms, GPU idle ~5 -> 1.6 ms.

### Reserve 7 instead of 10

The 10 GB reserve was sized for DFlash2's unbudgeted draft KV; MTP has none.
Reserve 7 boots (observed free 7.5 GiB against a 5.6 minimum) and keeps
**2427 hot experts per rank instead of 2284** (+143).

## Greedy A/B: `[8]` against `[1, 8]` (jobs 2096692, 2096693)

`ab.sh` + `bench.py`: two nodes, arms in opposite orders, server restarted per
arm; per arm three greedy runs of ~1450 tokens on the short and the 96K
prompt, step time from the engine counters over the decode.

**Greedy decoding is not reproducible here** (already noted for GLM-5.3 in
`2026-08-29-glm53-w4a16`): within one arm tokens per step ranges 4.0-7.0. And
step time *tracks* it, so arms only compare at matched acceptance. Least
squares over all 24 runs, `step_ms ~ tokens_per_step + arm + node + context`:

| term | ms/step |
|---|---|
| **`[1, 8]` vs `[8]`** | **-0.65 +- 0.28** |
| per accepted token per step | **+2.02 +- 0.19** |
| 96K vs short | -1.80 +- 0.30 |
| node 2096693 vs 2096692 | +0.45 +- 0.28 |

- Capturing the draft-decode graph is real but small, ~1.4%. The profile's
  ~5 ms of drafting idle was mostly profiler cost on ~430 eager launches;
  unprofiled, the host launches ahead of the GPU. Profiled idle time for
  eager code is not a usable estimate here.
- **Open: step time rises ~2 ms per extra accepted token.** The verify is
  always 8 tokens and the draft always 7 passes, so something per accepted
  token -- host-side output handling, or routing that differs with the
  content that accepts well -- costs ~2 ms. Not yet explained; it is larger
  than any other effect in the table.
- The earlier one-kernel vs Marlin comparison used single sampled windows and
  is inside this spread; it needs the same treatment before being quoted.

## Cold imbalance: activating the replicas (jobs 2097277, 2097278)

The production profile `glm53-w4a16-2496.json` already carries **985 Grace
replicas per rank**; the launchers leave assignment off. The earlier campaign's
OOMs with replicas came from PyTorch's pinned allocator rounding every cold
tier up to a power of two; cold tiers are now pinned with `cudaHostRegister`
(the MiMo work), so replicas cost their logical size (20.3 MiB each). The
campaign's pinned-rounding accounting patch is therefore obsolete and was not
applied: it would charge phantom bytes and refuse valid plans.

Offline first (`glm_replicas.py`, 5,173 held-out 8-position steps, units are
active cold experts on the busiest rank summed over layers):

| replicas | busiest rank | mean rank | excess |
|---|---|---|---|
| none | 354.8 | 226.7 | 128.0 (~7 ms at ~55 us per cold expert) |
| profile's 985 | 276.0 | 226.7 | 49.3 |
| greedy 1500 | 266.6 | 226.7 | 39.8 |
| greedy 2000 | 263.7 | 226.7 | 37.0 |

Served, two nodes, arms in opposite orders, greedy, fitted at matched
acceptance (`fit_ab.py off rA rB`, 48 runs, residual sd 0.69 ms):

| vs replicas off (MTP7, [1, 8], reserve 7) | ms per step |
|---|---|
| exact replicas, Triton assignment, Marlin | **-4.09 +- 0.25** |
| **exact replicas + INT4 one-kernel, balanced by time in the kernel** | **-5.51 +- 0.25** |
| node 2097278 vs 2097277 | +0.16 +- 0.20 |

The offline count model predicted ~-4.3 ms for exact; the one-kernel path adds
~1.4 ms on top even with MiMo's cost table (GLM's per-layer loads sit inside it;
INT4 experts are ~6% larger).

**The old TTFT blocker was a measurement artifact.** With back-to-back requests
the short-prompt TTFT is bimodal (0.195 / 0.235 s) and replicas land in the slow
mode more often; with a 0.5 s pause between 30 samples per arm it is 192-194 ms
off against ~196 ms exact, i.e. ~+3 ms, and the first request after boot is
slow in every arm. (Prefill is not the focus now; recorded for the rollout.)

Grace headroom, measured per NUMA node in the off arm: ~56 GiB anonymous (cold
tier ~47 + worker), ~45-50 GiB reclaimable page cache, 8-9 GiB free of 119 --
room for roughly 2,000-2,400 replicas per rank. 985 vs 2000 is measured next
(`glm53-w4a16-2496-r2000.json`, greedy minmax-cold on the training split).

## Verify-step GEMMs at M=8 (`bench_gemm.py`, bench-gemm-2097277.txt)

Per rank the verify step reads ~10.9 GB of bf16 weights -- more than the 7.8 GiB
an all-sharded count gives, because `fused_qkv_a_proj`, the router and the
indexer projections are replicated. Plain `F.linear` at M=8, 64 calls in a CUDA
graph over rotating weights, streaming read 3.65 TB/s:

| projection | us | floor | eff. | x/step | ms/step (floor) |
|---|---|---|---|---|---|
| o_proj 4096x6144 | 17.8 | 13.8 | 78% | 78 | 1.38 (1.08) |
| fused_qkv_a 6144x2624 (replicated) | 13.5 | 8.8 | 66% | 78 | 1.05 (0.69) |
| shared gate_up 6144x1024 | 7.8 | 3.5 | 44% | 75 | 0.58 (0.26) |
| shared down 512x6144 | 4.1 | 1.7 | 42% | 75 | 0.31 (0.13) |
| q_b 2048x4096 | 7.2 | 4.6 | 64% | 78 | 0.56 (0.36) |
| indexer wq_b / wk (replicated) | 7.6 / 5.6 | 4.6 / 0.5 | 61% / 10% | 21 | 0.28 (0.11) |
| lm_head 6144x38720 | 140 | 131 | 93% | 1 | 0.14 |

The fp32 router row of the raw file (19.5 us, 9%) is not what serving runs:
`GateLinear` sends M <= 16 to the cute-DSL `ll_bf16` GEMM, ~4 us x 75 in the
trace. Realistic GEMM headroom is ~1.2-1.8 ms/step, mostly the small-N shared
expert, the replicated qkv_a and o_proj.

## Attention: 16 heads padded to 64

`FlashMLASparseImpl`: the FP8 sparse decode kernel supports only h_q = 64 or
128, so each GPU's 16 heads are padded to 64 -- four times the attention work.
The 8-token verify reads only ~10.7 MB of KV per layer (~0.25 ms/step at HBM
speed) yet the kernel takes ~1.4 ms plus a 0.6 ms combine. Under DCP4 the
queries are all-gathered to 64 real heads and each GPU reads a quarter of the
(still on-GPU) KV, which also frees ~16 GiB per GPU for hot experts. Being
measured (`ab-dA`).

## DCP4 at c=1, and 985 vs 2000 replicas

**DCP4 vs DCP1** (job 2097277, 24 runs, both with exact replicas and the INT4
one-kernel path): **-1.12 +- 0.25 ms/step**. The KV cache stays on the GPUs,
sharded four ways (400,255 tokens), which lets the planner keep **3177-3239
hot experts per GPU against 2393-2466**, and the FP8 sparse decode kernel sees
64 real heads instead of 16 padded to 64. The gain is below what residency plus
padding predict (~3-4 ms), so DCP's per-layer collectives are taking most of
it back; profiled next (`run-dcp4-2097699`). **Adopted as the baseline:
`serve.sh` now defaults to DCP4, exact replicas, the one-kernel path, reserve 7,
capture [1, 8].**

**985 vs 2000 replicas** (job 2097278, 32 runs, one-kernel path):
2000 with the old slot +0.19 +- 0.36, 2000 with the fixed slot -0.66 +- 0.36,
a repeat of 985 -0.33 +- 0.36 -- about -0.5 ms, not significant, as the
offline count model predicted. Staying at the profile's 985.

## The prefill staging slot was sized with the replicas (vLLM 7b2dad1b41)

The cold prefetch slot held a layer's whole cold tier, own cold experts plus
the replicas stored after them: 830 MiB without replicas, 1478 with 985, 2572
with 2000. Prefill never reads replicas (static maps exclude them, and a staged
step is always above the replica-assignment token limit), so that was HBM taken
from hot residency: 2427 -> 2393 hot experts at 985, 2339 at 2000. Now the
planner budgets and the coordinator stages only a layer's own cold experts,
per component of the component-major tier. Served with 985 replicas: slot 790
MiB, 2427 hot, and `VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY=1` found 0 mismatches
over 370 staged layers of a 96K prefill. The prefetch GPU tests, broken since
cold tiers moved to `cudaHostRegister` (freed-but-registered test allocations),
are fixed and pass.

## DCP4 collectives: two cheap options tried (job 2097699)

DCP4 costs ~2.35 ms/step in 255 small NCCL collectives per step, each at the
~7-12 us floor of NCCL's LL ring: per layer a query all-gather (11.5 us), an
LSE all-gather (6.7 us) and an output reduce-scatter (8.6 us), plus 21 indexer
all-gathers (`breakdown-dcp4-2097699`).

- `--dcp-comm-backend a2a` (one all-to-all of outputs + LSE instead of the LSE
  gather and the reduce-scatter): **-0.21 +- 0.17 ms/step** over 32 runs, not
  significant. Default stays `ag_rs`.
- `VLLM_USE_NCCL_SYMM_MEM=1` (NCCL 2.28.9 symmetric kernels for dim-0 gathers
  and reduce-scatters): the server dies in CUDA graph capture with
  `cudaErrorStreamCaptureInvalidated`. Not usable on this stack.

Next: one-shot IPC all-gather / reduce-scatter kernels alongside vLLM's custom
all-reduce (same buffer registration and barriers; the TP group's instance
covers the same four GPUs), est. -1.2 ms/step.

## GSM8K, new baseline against the old configuration (job 2097700)

Paired, the same 1,000 questions, `gsm8k_paired.py`, c=1:

| | accuracy |
|---|---|
| new: DCP4, 985 replicas, INT4 one-kernel, [1, 8], reserve 7 | 0.9120 |
| old: DCP1, no replicas, Marlin | 0.9140 |
| difference | **-0.20 pts, 95% CI [-1.40, +1.00]**; 18 new-only, 20 old-only |

Symmetric discordance: noise. The earlier campaign's non-inferiority bar
(-1.0 pt lower bound) is not met at 1,000 questions; the full 1,319 would be
needed to claim it.

## Why step time rises with accepted tokens (steptrace-2097699)

`step-trace.patch` (env-gated, not committed to vLLM) logs each scheduler
update's time and accepted count; `steptrace.py` over 2,441 greedy steps:

| accepted in the step | 0 | 1 | 3 | 5 | 7 |
|---|---|---|---|---|---|
| median step, ms | 36.0 | 36.7 | 37.3 | 38.2 | 39.7 |

Regressed on the timed step's acceptance and its neighbours: 0.32 / 0.28 /
0.18 ms per token (the timed step, the one before, the one before that; with
async scheduling a scheduler gap times the next step). 32-step rolling means:
1.07 ms per token, correlation 0.76. The cost follows the text regime more than
the step: stretches of predictable text both accept well and verify slower,
most likely because 8 coherent tokens route to more distinct (cold) experts
than a batch the target rejects early. Not a fixed per-token cost to remove;
it means A/Bs must keep comparing at matched acceptance, and real-traffic
gains (lower acceptance) come out somewhat below the greedy bench's.

## One-shot DCP collectives (vLLM 82670c0023)

The custom all-reduce's one-shot algorithm, reused for DCP's all-gathers and
reduce-scatters: a JIT extension built against `csrc/custom_all_reduce.cuh`
takes the TP group's live `CustomAllreduce` by pointer (DCP4 = the TP GPUs), so
it shares its IPC buffer, CUDA-graph buffer registration and barrier flags.
`VLLM_DCP_ONE_SHOT_COLLECTIVES=1`; unsuitable inputs fall back to NCCL.

- correctness: `test_one_shot_collectives` (TP 2 and 4, eager and captured with
  replays, interleaved with all-reduces) and `test_one_shot.py` here (DCP shapes):
  gathers bit-exact, reduce-scatters equal to the reference
- microbenchmark: 78 query gathers [8, 16, 576] bf16 in a graph, **6.7 us each
  against NCCL's 13.8**
- end to end (`ab-oC`, `ab-oD`, 32 runs, two nodes, opposite orders):
  **-3.07 +- 0.20 ms/step** -- more than the collectives' kernel time in the
  profile (~2.35 ms), so NCCL's per-call cost outside its kernels was being paid
  too. Now a `serve.sh` default.

## Dense GEMMs: a bf16 weight-streaming kernel, parked

Production's dense GEMMs cost ~5.8 ms/step (oneshot-2097699, PDL off) against a
~3.0 ms byte floor: 154 calls/step of a 64x8 `nvjet` tile at 13.2 us, the
shared expert on split-K plus a separate reduce, Triton templates at ~7.5 us.
`vllm/model_executor/layers/skinny_gemm` (JIT, uncommitted) streams W through a
TMA ring over all SMs (stream-K, last-CTA-per-tile finish, no second launch).
Correct (error equal to cuBLAS's on every shape, deterministic), but no tile /
chunk / grid config beats `F.linear` except the tiny indexer `wk`
(`bench-skinny-2098311.txt`). ncu on the o_proj shape: cuBLAS 20.0 us at
2.66 TB/s (66% of DRAM peak), skinny 23.9 us at 2.23 TB/s, both ~14% occupancy
(`ncu-skinny-2098311.txt`). Even cuBLAS only reaches two thirds of the stream
rate at 50 MB per call; a perfect kernel would buy ~1.7 ms/step, a realistic one
~0.5-0.8. Parked in favour of the hot set below.

## The hot set under DCP4 was mostly arbitrary

When HBM outgrows a profile's hot list, `_promote_underfilled_residency` fills
the extra slots round-robin across layers **in expert-id order** ("ordering
within the promoted set carries no frequency information"). Under DCP4 that is
~715 of each GPU's ~3,211 hot experts. And the profile's own 2,496 were already
worse than plain frequency ranking.

Offline, held-out 8-position steps, 3,211 hot per GPU, 985 replicas:

| hot set | mean cold / GPU / step | busiest-GPU cold / step |
|---|---|---|
| profile + id-order promotion (today) | 152.3 | 201.6 |
| each GPU's most-used experts (training split) | **90.5** | **128.9** |

~73 fewer Grace reads on the critical GPU per step, ~-4 ms at ~55 us each.
`profiles/glm53-w4a16-freq-3250.json`: 3,250 hot per GPU (just above the largest
DCP4 budget), each layer's list in descending frequency so any demotion drops
the least-used first, 985 replicas recomputed for the new hot set; owners and
checkpoint fingerprint unchanged. Measuring in `ab-hP`.

## Full GSM8K, and how noisy a single run is (job 2098312)

All 1,319 questions, new baseline (DCP4, 985 replicas, INT4 one-kernel,
**one-shot collectives**) against the old configuration (DCP1, no replicas,
Marlin): 0.9098 vs 0.9212, -1.14 pts, naive paired 95% CI [-2.12, -0.15]
(15 new-only, 30 old-only). But the naive CI assumes a configuration answers a
question the same way every run, and greedy decoding here is not reproducible:
on the same 1,000 questions the **old configuration flipped 48 answers against
its own earlier run** (new: 33), more than new vs old (37).

Pooled over both runs of each arm on the shared 1,000 questions, bootstrapping
over questions: **-0.55 pts, 95% CI [-1.50, +0.40]**. The same bootstrap on old
vs old gives [-1.20, +1.50]: one paired run cannot resolve differences under
~1.3 pts here. No detectable regression; both runs lean slightly the same way
(and the two "new" runs differ by the one-shot collectives), so two more
repeats per arm are queued.

## The agentic profile, served (ab-gA, ab-gB; jobs 2100774, 2100775)

`profiles/glm53-w4a16-agentic-3239-r2000.json`, built from the MiMo-workload
capture (`../2026-09-28-glm53-route-cap`): 3,239 hot per GPU ranked by
frequency (runtime 3,208, so the planner only demotes the least-used), up to
2,000 replicas per GPU. Against the served `glm53-w4a16-2496.json` on the DCP4
baseline, four arms per node in opposite orders (gB's first new arm died on a
`profile_version` 1 / `secondary_ranks` mismatch, rerun as `new3`; gA's `new2`
JSON hit the project inode quota and was rebuilt from `ab-gA.log`):

    fit_ab.py --pool old gA gB      64 runs; residual sd 0.47 ms
      tokens/step      1.51 +- 0.08
      new vs old      -0.94 +- 0.12
      node gB          0.10 +- 0.12
      96k vs short    -1.59 +- 0.15

**-0.94 ± 0.12 ms per step** (by hand, ~-0.5 short, ~-1.0 at 96k). Offline it
predicts ~37 fewer cold experts on the busiest GPU (~2 ms at 55 us): the bench
text is not the capture workload, and the one-kernel overlaps part of the cold
reads with the hot ones. Unlike `freq-3250` (~0 on this bench), it wins on both
held-out workloads offline (`glm-profiles.png`), so it becomes the default.

## GSM8K, four runs per arm (gD, gF, gG, gH)

`gsm_pool.py gD gF gG gH`: every run of each arm pooled over the 1,000
questions all share, bootstrapped over questions. gG/gH (jobs 2101098/9) ran
the full 1,319 in opposite orders, `new` = DCP4 baseline with the served
2496 profile (gD's `new` predates the one-shot collectives).

    new: 0.9120, 0.9070, 0.9140, 0.9080; pooled 0.9103
    old: 0.9140, 0.9160, 0.9180, 0.9060; pooled 0.9135
    new - old: -0.33 pts, 95% CI [-0.92, +0.27]
    old against itself: 43 answers differ per pair of runs

No detectable regression, now bounded within about a point.
