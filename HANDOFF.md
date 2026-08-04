# GLM-5.2 on JUPITER: agent handoff

Last updated: 2026-08-04 (exact replica assignment shipped; tier-balance lever
closed; next lever is HBM residency)

## Start here

We are optimizing serving of `lowbitcoffee/GLM-5.2-W4A16` at up to 400K
context on one JUPITER Booster node. The implementation keeps the most
useful routed experts in HBM, runs the rest directly from NUMA-local Grace
memory through CUDA UVA, and uses the official GLM-5.2 MTP head for
speculative decoding.

The current qualified default is **MTP3 + concurrent hot/cold expert execution
under the tight Marlin shared-memory launch policy + exact replica assignment,
without sequence parallelism or local draft argmax**. Correctness remains
lossless because the W4 target model verifies the draft tokens.

Two serving configurations are qualified on one node:

- **Interactive**: c1/DCP1, V1 model runner, HBM main KV cache, 400K context.
- **Agent swarm**: c4/DCP4, V2 model runner, per-sequence KV, 11 GB planned
  HBM reserve.

Quote numbers with their harness attached; they are not comparable to each
other. The reference points are 37.57 tok/s for native vLLM CPU offload at
exact 400K, 123.65-127.67 tok/s decode for tiered MTP3 with overlap on that
same test, 108.08 decode tok/s on the warmed 24-prompt realistic suite at
c1/q4, and 181.60 tok/s aggregate at c4 on the 18-prompt mixed-domain suite.
The last two predate the shared-memory fix and the replica work respectively,
so both are conservative; no post-replica full-suite measurement exists yet.
`README.md` carries the full harness table and the lever ledger - read the
ledger before proposing an optimization, because most of the obvious ones are
already settled.

Source changes are pushed to the fork branch `tiered-moe-grace-mtp`; the human
must still review every changed line before proposing upstream work.

Read these before changing anything:

- [`AGENTS.md`](../AGENTS.md): mandatory vLLM development rules.
- [`README.md`](README.md): current state, lever ledger, and the chronological
  phase chain. The "Current state" section is the fastest way in.
- [`gh200-vllm-w4a16-tiered-moe-plan-v2.md`](gh200-vllm-w4a16-tiered-moe-plan-v2.md): implementation plan.
- [Replica scheduling v2](experiments/2026-07-31-replica-scheduling-v2/README.md): the most recent shipped change, and the measurement protocol every later A/B should copy.
- [Shared-memory monopoly](experiments/2026-07-29-marlin-smem-monopoly/README.md): the kernel bug that invalidated five earlier overlap experiments, plus the current step budget.
- [Tier balance](experiments/2026-08-01-marlin-tier-overlap/README.md): why the cold tier is retired as a target and what the next lever is.
- [MTP roofline](experiments/2026-07-18-mtp-prompt-profile/roofline-analysis.md): kernel, communication, and transfer analysis.

Keep changes minimal, measure every optimization against a matched control,
and update this worklog with commands, correctness evidence, and results. Log
each phase as it lands: a README phase entry, a dated experiment directory, and
a HANDOFF update, committed as you go. This file and the index drifted five
days and eight experiments behind between 2026-07-30 and 2026-08-04, and the
stale text was worse than no text - it sent readers to redo settled work.

## Repositories and state

The source repository is:

```text
/e/project1/profound/alint77/vllm
```

- Upstream: `vllm-project/vllm`
- User fork: `alint77/vllm`
- Branch: `tiered-moe-grace-mtp`
- Current commit: `cec73c66b` (`Load secondary expert copies only when
  assignment can use them`)
- Main implementation commit: `a66535e59`
- Branch base: `d08eebad1`
- No upstream PR has been opened.

Recent source history, newest first:

| Commit | What |
| --- | --- |
| `cec73c66b` | Load secondary expert copies only when assignment can use them |
| `44a24d909` | Read the tiered MoE config from its real field, enforce the layout |
| `919cfa9a7` | Graph-capturable route fingerprint for replica assignment |
| `ae4293d08` | Integrate exact replica assignment (v2) into the tiered MoE path |
| `53fe6dfcf` | Revert replica scheduling v1, **preserving** the shared-memory fix |
| `dc4dcff58` | v1 hardening; archived at tag `replica-scheduling-archive` |
| `1a94ed458` | v1 replica scheduling - also carried the Phase 26 shared-memory fix |
| `2f92b5365` | Avoid a blocking scalar H2D copy in DCP seq-len localization |
| `990b1d378` | Avoid NCCL symmetric all-reduce under DCP |

One process note worth not repeating: `1a94ed458` bundled the independently
qualified Marlin shared-memory fix into an unrelated feature commit, so when
that feature was reverted a plain `git revert` of either commit would have
taken the fix with it. `53fe6dfcf` handles it correctly and its message records
byte for byte what was preserved and that `tiered_moe_execution.py` was rebuilt
rather than reverted. Land qualified fixes as their own commits.

The vLLM source changes are committed and pushed. The outer worktree contains
untracked core dumps, Slurm logs, and the nested `agent_space` repository. Do
not use `git add -A` or delete those files without reviewing ownership.

This worklog is a separate Git repository nested at:

```text
/e/project1/profound/alint77/vllm/agent_space
```

- Remote: `https://github.com/alint77/jupiter-glm52-vllm-worklog`
- Branch: `main`
- Collaborator: `happykratos`

The worklog currently has pre-existing modified and untracked raw server,
profile, and Slurm outputs. Preserve them and stage files explicitly.

## Machine and scheduler

One Booster node provides:

- 4 NVIDIA GH200 Superchips and four 96 GiB HBM devices
- 4 Grace CPUs, 72 Arm cores each (288 total)
- about 857 GiB of NUMA-visible memory
- TP=4 and EP=4 for this model

Booster nodes have **no external internet access**. Download models, wheels,
and other artifacts on the login node before submitting a job. Model loading
uses ExaSTORE/GPFS; attempts to use `exa_fscratch`/ExaFlash were abandoned.

Each standard benchmark job requests one Booster node and four GPUs for four
hours. Up to eight nodes may be allocated concurrently, so independent test
matrix entries should normally be submitted in parallel. The existing job
template contains the correct account and partition:

```bash
sbatch agent_space/experiments/2026-07-18-mtp-fastpath/job.sh \
  3 false <unique-label> false
```

Use a unique label for every run because it names result files. Check jobs with:

```bash
squeue -u "$USER"
```

Jobs at the time of writing (2026-08-04): `1238882` is the `glm52-claude` c4
serving host on `jpbo-013-35`, which backs the local Claude Code launcher and is
not a benchmark - do not cancel it or read its metrics as an experiment.
`1239026` and `1239027` are unrelated four-node `arm9-mlm` jobs. Always re-check
with `squeue` rather than trusting this line.

## Environment

Always initialize the shell from the repository root:

```bash
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
```

That script loads the JUPITER 2026 stage, GCC 14.3, CUDA 13, CMake 3.31.8,
NCCL, ccache, and Ninja; activates `.venv`; selects the local model; redirects
compiler/runtime caches to scratch; and forces Hugging Face offline mode.

Do not use system `python3` or bare `pip`. Use `uv` for environment/package
operations and `.venv/bin/python` for Python commands. FlashAttention 3 was
installed from the CUDA 13/PyTorch 2.12 wheel index with:

```bash
uv pip install flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch2120
```

Important scratch locations:

```text
VLLM_CACHE_ROOT=/e/scratch/profound/naeimitabiei1/vllm-cache
CCACHE_DIR=/e/scratch/profound/naeimitabiei1/vllm-ccache
XDG_CACHE_HOME=/e/scratch/profound/naeimitabiei1/cache
FLASHINFER_WORKSPACE_BASE=/e/scratch/profound/naeimitabiei1/flashinfer
TRTLLM_DG_CACHE_DIR=/e/scratch/profound/naeimitabiei1/trtllm-deepgemm
```

Home storage has an inode limit of roughly 82K. `~/.cache/vllm` now points to
`/e/scratch/profound/naeimitabiei1/vllm-cache`; the existing cache was merged
and verified before the symlink replaced it. UV, Triton, Cargo, VS Code,
Cursor, W&B, and other large state are also symlinked out of home. Do not move
caches back into home.

## Local model artifacts

```text
../models/GLM-5.2-W4A16-55c92ae
../models/GLM-5.2-W4A16-FP8-MTP
/e/scratch/profound/naeimitabiei1/glm52-fp8-mtp-layer78
```

The first path is the immutable 361 GiB W4A16 target. The second grafts the
official FP8 layer 78 MTP weights onto it. Its eight target shards are hard
links, while the roughly 10 GB MTP delta is stored in three additional shards.
Do not modify either checkpoint in place.

The Booster runtime is offline, so serving must use these local paths rather
than Hugging Face identifiers.

## What was implemented

The cumulative branch adds:

1. A fail-closed GLM-specific physical planner for HBM/Grace expert placement,
   strict machine and NUMA validation, and a plan-only CLI path.
2. Checkpoint filtering by EP ownership before tensor payload reads.
3. Bounded conversion and direct loading into compact HBM or pinned-Grace UVA
   Marlin storage.
4. HBM and host-UVA MLA-cache planning, with HBM retained as the default.
5. Trace-driven expert ownership/placement and long-context capacity guards.
6. A mirrored batch-one target MoE path that avoids 150 sequence-parallel
   NCCL reduce-scatter/all-gather pairs per token.
7. Loading and planning for the grafted FP8 MTP layer with native EP4 expert
   ownership.
8. Hot-HBM/cold-Grace stream overlap for MTP verification sizes up to four.
9. An opt-in local draft-argmax reduction. It greatly reduces collective
   bytes but did not improve batch-one throughput reliably, so it is off.
10. A DCP port of the FlashMLA sparse backend, admitting DCP1/DCP4, with
    block-sharded KV planning and per-sequence provisioning at c4.
11. AutoRound W4G64 loading through the same tiered Marlin path.
12. A tight Marlin shared-memory launch policy (`VLLM_TIERED_MOE_TIGHT_SMEM`,
    hot at 2 CTAs/SM and cold at 1), which is what makes the two tiers actually
    co-resident.
13. Exact per-step replica assignment: selected routed experts hold a second
    physical copy in paired Grace memory, and a fused kernel assigns each active
    logical expert to one copy to minimize the predicted slowest-rank MoE time,
    with a graph-capturable route fingerprint enforcing exactly-once.

The main launch path is composed from:

```text
agent_space/experiments/2026-07-18-mtp-fastpath/run-server.sh
agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh
```

The qualified settings are MTP3, TP4/EP4, strict NUMA binding, FP8 DSA-MLA KV
cache, HBM main KV cache, 10 GB planned HBM reserve, Inductor mode 3, a size-4
full/piecewise CUDA graph, and `fuse_allreduce_rms=false`.

## Performance progression

All headline decode tests are batch one, greedy, ignore EOS, 256 output tokens.

| State | 4K decode | Exact-400K decode |
| --- | ---: | ---: |
| Native vLLM CPU offload baseline | 37.06 tok/s | 37.57 tok/s |
| Tiered implementation before MTP | 51.74 tok/s | 55.16 tok/s |
| Initial MTP3 graft | 103.42 tok/s | 108.17 tok/s |
| MTP3 with verification overlap | - | 127.67 tok/s mean |
| Fresh no-SP control after SP trial | - | 123.65 tok/s mean |

Later gains were measured on suites and on step time rather than on this
synthetic test, and are not additive with the rows above:

| Change | Harness | Effect |
| --- | --- | --- |
| DCP4 at concurrency 4 | exact 400K, c4 | 173-179 tok/s aggregate |
| V2 runner MTP full graphs | 4K c4 | 186.52 -> 225.23 tok/s effective |
| Draft-sync H2D fix | realistic c4 warmed | 179.47 -> 185.05 tok/s |
| Marlin shared-memory fix | no-MTP step time, c1/c2/c4 | -5.3% / -6.4% / -6.7% |
| Exact replica assignment | acceptance-free batch 4 | -5.96% step time |
| Exact replica assignment | MTP3 batch 16 | -6.50% step, -5.11% corrected |

The overlap result is about 18% faster than its matched serial MTP3 control.
Its exact-400K draft acceptance is about 60-61%. Across 18 Python, PyTorch,
CUDA C++, math, email, and explanation prompts, weighted acceptance was 67.61%
and weighted decode was 93.86 tok/s.

MTP2 was slower at roughly 110-114 tok/s. MTP6 improved the short-context
number slightly but fell to 84.47 tok/s at exact 400K because late draft
positions were rarely accepted. MTP7/8 were skipped. MTP3 remains the default.

## Correctness and profiling

Performance is not accepted from throughput alone. The qualified runs used:

- greedy deterministic decoding and an exact token/string comparison;
- a semantic smoke prompt whose eight-token continuation is
  ` Paris. Distance from Paris to Lyon is`;
- exact-400K continuation SHA-256
  `d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`, which
  predates the Marlin grid change and has to be re-established (see below);
- focused unit tests, project hooks, and source guards recorded in each report;
- repeat runs for optimization decisions.

Two later corrections to how correctness is gated:

- **Exact-text A/B between two servers is not a valid gate.** Off-versus-off
  matched 0 of 8 completions with a median first-token divergence of 3.5
  tokens. Gate kernel and scheduling changes on invariants instead:
  exactly-once execution, launch census, tolerance-bounded tensor compare.
- A grid change moves Marlin's fp32 reduction order by at most one bf16 ulp,
  deterministically. That changes the golden SHA without being a regression.

The roofline/profile showed that target routed W4 MoE, synchronization, and
dense W4 kernels dominate; the FP8 MTP block itself is small. Verification
overlap removed about 15.7% from the MTP3 routed span. Local draft argmax cut
the per-position TP4 ring payload from about 232 KB to 24 bytes but was not a
stable throughput win.

Target-only sequence parallelism was implemented experimentally and reverted.
It averaged 112.98 tok/s versus 123.65 tok/s for the fresh no-SP control. The
routed kernels did not get shorter, while each target MoE layer introduced a
reduce-scatter/all-gather pair.

## Known traps

- `fuse_allreduce_rms=true` caused an illegal memory access on this stack.
- `gpu-memory-utilization=0.94` left too little transient HBM for prefill;
  qualified runs use 0.90 plus the explicit tiered reserve.
- Ordinary pageable Grace allocations migrated into HBM. The implementation
  therefore uses NUMA-bound pinned Grace memory and validates locality.
- Host-UVA main KV cache improves expert residency and decode but badly hurts
  TTFT; AUTO remains fail-closed on HBM.
- Cold first startup can spend a long time reading GPFS and compiling
  FlashInfer. Reuse scratch JIT caches and keep `MAX_JOBS=4`; do not diagnose
  warmed request throughput from startup time.
- Parallel jobs need job-specific JIT/cache directories to avoid corruption;
  `mtp-fastpath/job.sh` already derives them from `SLURM_JOB_ID`. Per-job cache
  roots multiply the inode problem below, so prefer per-arm roots under
  `/e/project1/profound/alint77/.marlin-caches/`.
- Do not re-enable target sequence parallelism or deeper fixed MTP without a
  new measured reason. Wider third-party speculators are settled the same way:
  DSpark and DFlash both work and both lose, in strict inverse order of verify
  batch width.
- **Never compare an in-situ time against an isolated time and call the
  difference recoverable.** This error produced a confident wrong answer at
  least three times: the "39-40% overlap saving", the "12 ms/step of hot/cold
  overlap headroom", and the Grace-to-HBM staging win below.
- Identifying the hot and cold Marlin tiers by CUDA stream is invalid under
  graph replay, where stream ids rotate per layer. Identify them by grid: 264
  blocks hot, 132 cold.
- Timing a fork/join eagerly charges two stream barriers per iteration, a
  ~110 us floor on Booster that can exceed the whole kernel at low activated-
  expert counts. Measure under graph replay.
- The login node and Booster do not prefer the same Marlin grid (66 CTAs vs
  132). Never tune a shipped constant off Booster.

## Current thread: buy HBM residency (2026-08-01, open)

The tier-balance work closed the last cheap lever and named the next one.

Within a routed layer the two Marlin tiers now overlap essentially perfectly:
the measured union is 310.0 us against a 308.0 us floor set by the observed
chain lengths, and the streams start within 1.2 us of each other. Occupancy,
grid shape and register pressure have at most **0.15 ms/rank-step** left in
them. The union/sum ratio of 0.70 is not an overlap defect - it is that one
tier does about 39% more work than the other in the layer.

Redistributing HBM slots across layers to fix that imbalance models at
**exactly 0.0%**, because it only pays when some layer is hot-bound and none is.
A cold expert costs 45.32 us against a hot expert's 9.75 us, 4.6x, and 6.77 cold
experts per layer in 306.9 us is ~442 GB/s per rank - at or past the 421 GB/s
C2C roof. The cold tier is moving weights as fast as the link allows.

So the only remaining move on the routed MoE is to move fewer weights:

1. **Reclaim HBM.** `gpu_memory_utilization` is 0.85 and the tiered reserve is
   7 GB. Freeing ~5 GB safely is the cheapest thing to test and should be next.
   Balance would want ~4.7 GB more resident against a measured peak free of
   3,874 MiB, so this is close to the right size.
2. **Shrink the resident copy.** The cold tier moves 20.05 MB per expert; any
   reduction converts to cold-tier time at 4.6x the leverage of hot work.
3. **The sampling and logits boundary**, ~1.8 ms/step, is the largest remaining
   non-MoE item and is host-bound.

Do not spend time on: cold-tier kernel tuning, tier rebalancing, occupancy
work, green contexts, or capturing the draft graphs. All are measured and
closed - see the lever ledger in `README.md`.

Also open, and undocumented: `experiments/2026-08-01-marlin-grid-fit` holds raw
sweep results from jobs 1197398, 1197614, 1197769 and 1198412 with no report, an
empty `analysis/`, and nothing committed. `policy-1197614.json` reads as the
shipped 264/132 grid at 93.67 us against 129.91 us for the pre-fix launch, with
the best alternative found (264/99) 1.6% better over 51.6% of the c1 operating
point. Either finish the report or delete the directory; do not leave it as a
third state.

## Earlier thread: the Marlin shared-memory monopoly (2026-07-29, shipped)

Marlin's MoE launch passed `deviceSharedMemOptin / blocks_per_sm` as its dynamic
shared memory instead of the `sh_cache_size` it computes one line earlier, so a
single wave claimed ~100% of every SM's shared memory and **no second Marlin CTA
could be placed at any `blocks_per_sm`**. That is why every hot/cold overlap
experiment returned nothing: the tiers were serialized by the block scheduler
before any stream, priority, order or occupancy knob applied. Requesting only
what the kernel uses, and launching hot at 2 CTAs/SM and cold at 1, cuts the
measured two-tier union by 41-48% on the login node and lands it on
`max(hot, cold)`.

Source changes are committed and live at HEAD. They landed bundled inside
`1a94ed458` and survive via the hand-built revert `53fe6dfcf`: `ops.cu`,
`torch_bindings.cpp`, `_custom_ops.py`, `envs.py`,
`marlin_moe.py`, `tiered_moe_execution.py`, plus
`tests/kernels/moe/test_moe.py::test_fused_marlin_moe_launch_policy`. Defaults
are unchanged for every non-tiered caller, and `VLLM_TIERED_MOE_TIGHT_SMEM=0`
restores the old launch. `_moe_C_stable_libtorch.abi3.so` was rebuilt in place;
the pre-change build is saved at
`/e/scratch/profound/naeimitabiei1/_moe_C_stable_libtorch.abi3.so.bak-20260717`.

Qualified on Booster: the mechanism reproduces exactly (same shared-memory
geometry, per-CTA `%smid` peak of 3 at the production request vs 4 once it
fits), the isolated two-tier union drops 40%, and the full 15-cell
activated-expert sweep under CUDA-graph replay improves by a median 34% with
every cell positive and 31-45% at the production c1/q4 shape. A twelve-way grid
search confirms the shipped `{"hot": 2, "cold": 1}` is optimal on Booster - note
it is *not* what the login node prefers, so do not retune that constant off
Booster.

Open items:

- Server-level end-to-end is **qualified**, by three independent methods that
  agree in the -5% to -8% band:
  - Same-node c1/q4 MTP3 (1090344): 98.35 -> 105.28 output tok/s (**+7.05%**),
    TPOT 9.183 -> 8.436 ms, acceptance matched to 0.51% (2.9212 vs 2.9360 tokens
    per target step), so **step time falls 7.66%**.
  - Realistic agentic coding suite (1090343, 16 prompts of 131-2277 input
    tokens, 512 out): 94.13 -> **99.79 tok/s (+6.01%)**. Lower than the gate
    suite because longer inputs dilute the routed-MoE share.
  - No-MTP, acceptance-free (1090345), where TPOT *is* step time: **-5.28% at
    M=1, -5.80% at M=2, -6.66% at M=4**, spreads +-0.1 ms. The gain rising with
    M is what the mechanism predicts.

  The earlier cross-node +9.71% was inflated by about 2.5 points of node and
  acceptance luck; **~+7% is the honest figure**.

- The exact-400K golden SHA is expected to change: the shipped grid moves work
  between Marlin's data-parallel and split-K halves, so the fp32 reduction order
  differs by at most one bf16 ulp, deterministically. Re-establish it; do not
  treat a mismatch as a regression. **Still not re-established as of
  2026-08-04**; the replica work landed on top of it, so re-establish the SHA
  against current HEAD rather than against the Phase 26 tree.

### Where the decode step goes now, and what is retired

A matched off/on profiler capture on two nodes (`job-profile-ab.sh`, jobs 1092954
and 1092955) was re-analysed with launch-correlation attribution
(`analyze_step_budget.py`), which makes the per-step kernel census exact in all
176 rank-steps: 166 custom all-reduces, 306 Marlin GEMMs, 4 all-gathers. The
earlier GPU-timestamp-versus-CPU-annotation method dropped 4-6% of kernels. Share
percentages survived the correction to within 0.5 points; absolutes are ~6%
higher and sum to the 25.851 ms attributed GPU span, not the 24.307 ms step wall
(consecutive steps' GPU spans overlap by 1.54 ms).

Three results should shape what gets tried next:

- **A quarter of every step does no useful work.** 2.01 ms GPU-empty plus 4.13 ms
  of all-reduce barrier spin that the profiler scores as busy = 6.14 ms of
  24.307 ms. Utilisation is ~75%, not the 92% the busy union suggests.
- **The cold tier is at the C2C roofline and is retired as a kernel target.**
  53 us/layer for a 20.1 MB W4G64 expert is 379 GB/s against the 373 GB/s
  measured achievable rate (`2026-07-25-grace-bandwidth`). It is node-invariant
  to 0.2% where hot varies 2.5%. Do not tune cold tiles, split-K, grids or
  occupancy. Residual overlap headroom was bounded here at 1.72 ms/step and
  Phase 32 later tightened it to 0.15 ms/rank-step.

  On Grace-to-HBM staging, this line previously read "refuted twice already".
  That is no longer the whole story: the 2026-07-27 green-context experiment
  measured a within-layer staged path 19.8% faster at m=16 and 36-38% at m=256,
  bit-exact for the copy. But its production control was the **pre-fix** co-run,
  which the shared-memory monopoly had serialized, and the premise of staging is
  "stop co-running" - which the fix already achieved. Treat staging as neither
  refuted nor open: it is unmeasured against current production, and any retry
  starts by re-running that control.
- **Per-layer EP-rank balance was the top lever, at 3.615 ms/step** of summed
  `max - mean` routed-layer skew, 47% of the routed span, and it is now the
  shipped replica assignment. This corrected an earlier 1.485 ms figure and
  agrees within 12% with the 4.13 ms of all-reduce synchronization excess:
  **one lever, not two additive ones.** Worst layers were 34, 69, 18, 20, 61,
  67. Phase 31's paired trace took skew from 8.230 to 3.170 ms/rank-step
  (-61.5%) and all-reduce residency from 9.223 to 4.114 ms (-55.4%), worth 5-6%
  end to end. What remains of this bucket is the per-step max-of-four-ranks
  order statistic, which no assignment can remove.

And on the GPU-empty half, from `analyze_graph_gaps.py`:

- **The MTP drafts are already CUDA-graphed.** Seven graphs replay per step: the
  target at 3,466 kernels plus a 16- and a 20-kernel graph per draft round. The
  six draft graphs hold 0.003-0.008 ms of internal idle each. Do not propose
  capturing the draft path.
- **The sampling/logits boundary after the target graph is 0.745 ms of device
  idle** plus 1.071 ms of un-graphed GPU work, 0.561 ms of which is the
  `[4, 6144] x [6144, 38720]` vocabulary projection. That is two thirds of all
  between-graph idle in one of seven boundaries. For 0.703 ms of it the host is
  inside no CUDA call, so it is Python sampler work: **graph capture alone will
  not recover it**, and the fix has to reduce or overlap host work.
- In-graph idle is 1.019 ms across ~2,430 sub-microsecond dependency gaps
  (p50 0.42 us, max under 1.8 us). Graph-node fusion caps out around 1 ms and is
  a secondary lever.

Do not use the trace's -4.56% as the size of the shared-memory win; it is c1/M=4
under an active profiler on one prompt. The no-MTP suite is the authority
(-5.3% to -6.7%, growing with concurrency), and the trace explains where.

**Scratch is inode-limited, not space-limited.** Three jobs died in Inductor
autotuning with `Errno 122` while 512 MB writes still succeeded: only five files
could be created on `/e/scratch` against 4000/4000 on `/e/project1`. The shared
`vllm-cache` root was the consumer and was cleared (symlink preserved). Per-job
cache roots multiply the problem and were replaced by per-arm ones; the jobs in
this experiment now cache under `/e/project1/profound/alint77/.marlin-caches/`.
The many empty `vllm-cache-<JOBID>` leftovers are a red herring - they hold one
inode each.

See [the shared-memory monopoly](experiments/2026-07-29-marlin-smem-monopoly/README.md).

## Checkpoint loading: exa_fscratch is 20x faster under concurrency

Checkpoint load is the slowest part of every job here (86 shards of 5.37 GB,
~10.5-11.8 s each, ~15 minutes). Measured on Booster (job 1091250,
`analysis/fs_bench.py`), `exa_fscratch` against `exa_project1` where the models
currently live:

| access pattern | project1 | fscratch | ratio |
| --- | ---: | ---: | ---: |
| O_DIRECT QD1 | 0.14 GB/s | 0.34 | 2.5x |
| buffered sequential | 2.79 | 7.94 | 2.8x |
| **O_DIRECT x4** | **0.55** | **11.00** | **20x** |
| **O_DIRECT x8** | **0.95** | **21.37** | **22x** |
| mmap page-fault | 3.13 | 5.12 | 1.6x |

The load-bearing row is parallel O_DIRECT: fscratch scales with concurrency
(11 -> 21 GB/s) while project1 barely does (0.55 -> 0.95). That is exactly the
production pattern - four ranks reading at once. project1 is latency-bound at
shallow queue depth and only reaches bandwidth through readahead, which is why a
streaming `cp` runs 4x faster than the loader.

**Do not quote the `safetensors` row from that job.** It read 0.10 GB/s on
project1 against a real loader that achieves ~0.5 GB/s, because the benchmark
calls `get_tensor` per key and serialises small reads. Its 57x ratio and the
"78 min -> 1.4 min" extrapolation the script printed are both artifacts of a
pessimal access pattern; only the end-to-end load test settles the real number.

`exa_fscratch` is 15 PB at 2% used and writable at over 2 GB/s. Note the
baseline README records that ExaFlash/`exa_fscratch` was investigated and
abandoned early in the project - check whether that was a retention or capacity
policy rather than a performance result before relying on it, since a filesystem
that purges would make this a per-job staging step rather than a new home for
the checkpoints.

## Earlier thread: DCP4 and concurrency 4

Read, in order: the [SOL re-analysis](experiments/2026-07-18-sol-reanalysis/README.md)
(kernel-level roofline of the qualified config, TP2xPP2 rejection, DCP
rationale, c=4 KV math) and the
[DCP port](experiments/2026-07-18-dcp-port/README.md) (implementation notes,
the lse=+inf trap, round-1 results, in-flight jobs).

State at last update: the base DCP thread is complete and qualified. The
serving menu on one node (all lossless, golden-SHA-gated): c=1 DCP1 at
129-136 tok/s (interactive default); c=4 DCP4 at 173-179 tok/s aggregate
(43-45 tok/s per agent) for the 400K agent swarm, launched via
`job-c4.sh 3 4 4 <label>` / `run-server-c4.sh`. The full-graph capture
crash was root-caused to the graph-memory profiling pass (temporary pool +
minimal KV cache) and fixed by skipping FULL-graph profiling under DCP.
Known operational notes: free HBM varies ~0.5 GiB across nodes (one audit
failure at 10 GB planned reserve; 11 GB is safe for c=4), and 400K
prefills serialize at ~110 s each (cold simultaneous agent starts ladder
TTFTs; cross-turn prefix caching mitigates). A2A+NVLS reached 190.15 tok/s
once but is experimental: a replica hung in collective symmetric-memory
registration. Commit `990b1d378` disables symmetric-memory all-reduce under
DCP while retaining NVLS AG/RS; it passed commit hooks but still needs one
runtime qualification. No Slurm jobs are active. The stable DCP4 default is
still `ag_rs` at 179.56 tok/s aggregate.

## Settled: third-party speculators lose to MTP3 (2026-07-25)

This section previously recommended downloading and evaluating DFlash. **That
work is done.** Both DFlash and DSpark were downloaded, made to run on this
stack, and rejected on measurement. Do not redo it.

| | MTP3 (V2, c1) | DSpark t=8 | DFlash t=15 |
| --- | ---: | ---: | ---: |
| verify batch | 4 | 9 | 16 |
| acceptance length | ~2.9 / 4 | 3.98 / 8 | **6.84 / 15** |
| implied decode tok/s | **106.08** | ~93 | ~70 |
| realistic end to end | **95.44** | 84.67 | 65.6 |
| implied step time | ~27 ms | ~42 ms | ~97 ms |

Both reproduce the exact deterministic completion, so neither failed on
correctness. **The ranking is strictly inverse to verify-batch width, and
acceptance length is anticorrelated with throughput**: DFlash accepts more than
twice as many tokens per step as MTP3 and is 34% slower end to end. The cause
is structural, from the critical-path review - this target's routed MoE is only
35% fixed weight streaming and 65% proportional to token count, so verification
cost grows nearly linearly with block width while acceptance grows sublinearly.

The consequence generalizes: **any wider speculative block loses here unless it
also cuts per-token routed-MoE cost.** Do not evaluate another wide-block
speculator on acceptance figures from its model card.

Source changes for both were reverted. DSpark's bring-up cleared five real
defects worth knowing about if a similar draft is ever attempted - three
upstream (`642076d26`, `a7d00ec05`, `e18f0037a`) and two in this branch's
tiered contract (the dtype check rejecting a draft-derived `VllmConfig`, and
tiered KV allocation assuming MLA-only cache specs). Details in the
[DSpark bring-up](experiments/2026-07-25-dspark/README.md) and
[DFlash result](experiments/2026-07-25-dflash/README.md).

FlashMLA-ETAP remains uninvestigated.

## First-session checklist

```bash
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
git status --short --branch
git -C agent_space status --short --branch
squeue -u "$USER"
```

Then read the latest experiment report, state the control and correctness gate
before coding, and submit independent jobs in parallel. Record results in a
new dated directory under `agent_space/experiments/`, commit only intended
files in the nested worklog, and push source changes to the fork branch after
the human has reviewed every changed line.
