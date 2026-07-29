# GLM-5.2 on JUPITER: agent handoff

Last updated: 2026-07-29 (Marlin shared-memory monopoly; qualification in flight)

## Start here

We are optimizing serving of `lowbitcoffee/GLM-5.2-W4A16` at up to 400K
context on one JUPITER Booster node. The implementation keeps the most
useful routed experts in HBM, runs the rest directly from NUMA-local Grace
memory through CUDA UVA, and uses the official GLM-5.2 MTP head for
speculative decoding.

The current qualified default is **MTP3 + concurrent hot/cold expert
execution, without sequence parallelism or local draft argmax**. It delivers
about 124-136 decode tok/s on the exact 399,744-input/256-output test, versus
37.57 tok/s for native vLLM CPU offload. Correctness remains lossless because
the W4 target model verifies the draft tokens.

Phase 15 is a decode-context-parallel (DCP4) port of the FlashMLA sparse
backend for concurrency-4 serving at 400K. DFlash was deprioritized: the
draft is only 7% of the step and DFlash's ~3K-context training makes 400K
acceptance a poor bet. Source changes are pushed to the fork branch
`tiered-moe-grace-mtp` through `990b1d378`; the human must still review every
changed line before proposing upstream work. See the
[DCP port experiment](experiments/2026-07-18-dcp-port/README.md) for state,
including any still-running Slurm jobs.

Read these before changing anything:

- [`AGENTS.md`](../AGENTS.md): mandatory vLLM development rules.
- [`README.md`](README.md): chronological project status and headline results.
- [`gh200-vllm-w4a16-tiered-moe-plan-v2.md`](gh200-vllm-w4a16-tiered-moe-plan-v2.md): implementation plan.
- [MTP fast-path report](experiments/2026-07-18-mtp-fastpath/README.md): current result and rejected sequence-parallel trial.
- [MTP roofline](experiments/2026-07-18-mtp-prompt-profile/roofline-analysis.md): kernel, communication, and transfer analysis.

Keep changes minimal, measure every optimization against a matched control,
and update this worklog with commands, correctness evidence, and results.

## Repositories and state

The source repository is:

```text
/e/project1/profound/alint77/vllm
```

- Upstream: `vllm-project/vllm`
- User fork: `alint77/vllm`
- Branch: `tiered-moe-grace-mtp`
- Current/pushed commit: `5000658c4` (`Optimize tiered MoE MTP verification`)
- Main implementation commit: `a66535e59`
- Branch base: `d08eebad1`
- No upstream PR has been opened.

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

There were no active jobs when this handoff was written.

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
  `d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`;
- focused unit tests, project hooks, and source guards recorded in each report;
- repeat runs for optimization decisions.

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
  `mtp-fastpath/job.sh` already derives them from `SLURM_JOB_ID`.
- Do not re-enable target sequence parallelism or deeper fixed MTP without a
  new measured reason.

## Current thread: the Marlin shared-memory monopoly (2026-07-29)

Marlin's MoE launch passed `deviceSharedMemOptin / blocks_per_sm` as its dynamic
shared memory instead of the `sh_cache_size` it computes one line earlier, so a
single wave claimed ~100% of every SM's shared memory and **no second Marlin CTA
could be placed at any `blocks_per_sm`**. That is why every hot/cold overlap
experiment returned nothing: the tiers were serialized by the block scheduler
before any stream, priority, order or occupancy knob applied. Requesting only
what the kernel uses, and launching hot at 2 CTAs/SM and cold at 1, cuts the
measured two-tier union by 41-48% on the login node and lands it on
`max(hot, cold)`.

Source changes are in the working tree (not yet committed at the time of
writing): `ops.cu`, `torch_bindings.cpp`, `_custom_ops.py`, `envs.py`,
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
  treat a mismatch as a regression.
- Nothing is committed. Source changes sit in the working tree.

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
  occupancy - and do not re-try Grace-to-HBM staging, refuted twice already.
  Residual overlap headroom is a bounded 1.72 ms/step (span above
  `max(hot, cold)`).
- **Per-layer EP-rank balance is the top lever, at 3.615 ms/step** of summed
  `max - mean` routed-layer skew, 47% of the routed span. This corrects an
  earlier 1.485 ms figure in the report and now agrees within 12% with the
  4.13 ms of all-reduce synchronization excess: **they are one lever, not two
  additive ones.** Worst layers are 34, 69, 18, 20, 61, 67. The 4.13 ms is a
  ceiling (per-call min over four ranks, unachievable by any single rank); treat
  3.6 ms as the target.

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

## Earlier thread: DCP4 and concurrency 4 (supersedes DFlash)

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

## Superseded next experiment: DFlash

DFlash and FlashMLA-ETAP were investigated but have not been implemented or
benchmarked in this project. No DFlash checkpoint has been downloaded yet.

The relevant draft is `UCloud-org/GLM-5.2-FP8-DFlash`. It is a five-layer,
block-size-16 DFlash model trained against GLM-5.2-FP8. The current vLLM tree
already contains native DFlash support. Because our verifying target is W4A16,
output correctness should remain lossless, but acceptance and speed must be
measured rather than inferred from the checkpoint's FP8-target results.

Recommended next steps:

1. Download the roughly 7.5 GB DFlash checkpoint on the internet-facing login
   node into a new immutable local model directory.
2. First run a short-context load/correctness/capacity smoke on one Booster
   node; do not start with 400K.
3. In parallel, run a matched 4K prompt-suite comparison against the current
   MTP3 default and a 400K capacity/startup test.
4. The draft KV cache is the main long-context risk. Estimate per-rank draft KV
   at about 7.6 GiB in BF16 or 3.8 GiB in FP8 at 400K, then confirm actual
   allocations. Use FP8 draft KV if the vLLM configuration supports it.
5. Only profile DFlash after it passes exact-output checks and beats or
   plausibly approaches MTP3 on the matched benchmark.

The DFlash checkpoint was trained mostly on English, non-thinking data around
3K context. Expect prompt- and length-dependent acceptance, especially at
400K. Keep DFlash changes separate from the already-qualified MTP3 path.

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
