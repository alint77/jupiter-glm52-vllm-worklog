# GLM-5.3 decode placement and replica assignment

Status: GPU serving qualification in progress. Production rollout is authorized for improvements that
pass correctness and latency qualification. No candidate has qualified yet.

## Objective

Reduce the sum of per-layer cross-rank completion times during decode, and
improve Claude Code time per output token without degrading model quality,
prefill latency, or supported context/concurrency. Optimize HBM residency and
Grace replica selection together, using the actual planner result.

The current profile is already labelled `replicated-makespan-v1`; this is an
evaluation and update of that machinery, not a new frequency-versus-latency
invention. The GLM-5.3 launchers explicitly left replica assignment off because
it had only been qualified on GLM-5.2. The existing fused exact scheduler pins
hot experts to their owners and minimizes maximum active cold-expert count.

## Sequence and gates

1. Record the current code, profile, routing-data split and production shapes.
   Run the existing GPU replica correctness suite and a GLM-5.3 exact startup
   and routing-invariant smoke test in a separate Slurm allocation.
2. Replay the real Claude Code routing capture, separating requests before
   sampling decode verification groups. Compare current/off, current/exact,
   refreshed/off and refreshed/exact at deployed residency. Preserve an unseen
   request holdout. Count distinct active experts, not token frequency alone.
3. Build joint residency/secondary candidates on training data only. Minimize
   the layer maximum after exact assignment; respect per-rank HBM and Grace
   budgets. Report the model as a proxy until calibrated against this model's
   trace. Do not transplant GLM-5.2 microsecond constants as measurements.
4. Same-node, order-balanced serving A/B with fixed prompt/output lengths,
   warmups, multiple paired rounds and saved raw responses/metrics. Separate
   unprofiled TTFT/TPOT, speculative acceptance and engine-step evidence.
   Start with c1/MTP3/DCP1 (the reviewed trace), qualify c4/MTP3/DCP4 separately,
   and do not transfer a result to DFlash2 without testing that shape.
5. Validate routing and tensor invariants, plus paired model evaluation.
   Nonsignificant accuracy differences are not proof of equivalence. Record
   paired confidence intervals and an explicit noninferiority margin.
6. Profile the winning pair to verify reduced per-layer cold-work imbalance
   and collective waiting. Check short and long prefill and interaction with
   cold prefetch if it becomes enabled during this campaign.
7. Enable only qualified profile/assignment combinations in their applicable
   production scripts, preserving a documented rollback and recording hashes.

## Controls

- Existing dataset: `/e/fscratch/profound/naeimitabiei1/caches/routes/snap-1535650-a`.
- Existing profile: `agent_space/profiles/glm53-w4a16-2496.json`.
- Prior implementation/results: `../2026-07-31-replica-scheduling-v2/`.
- Decode mechanism: `../2026-09-04-mtp3-profile/DECODE-REVIEW.md`.
- Cold prefetch is now enabled in DFlash2 production at 1,024 tokens. Final
  qualification includes that setting; do not modify its campaign's results.
- Use isolated job/result paths and explicit flags for all benchmark arms.
- Keep existing production servers running while testing.

## Activity

- 2026-09-05: experiment created; audited launchers and existing solver.
  Current GLM-5.3 MTP3 and DFlash2 scripts omit replica assignment. The stored
  2,496-hot-slot profile contains 3,940 secondary candidates; assignment off
  means those copies are not allocated. Actual hot residency varies with the
  serving shape, HBM reserve and prefetch staging budget.

- Frozen source: `provenance.json` (MTP3), `qualification-provenance.json`
  (DFlash2). Detached worktrees prevent concurrent source edits changing arms.
- Fixed a real correctness bug before benchmarking exact assignment:
  `attach_tiered_moe_layer_placement` used the profile's nominal hot set even
  when the planner demoted/promoted experts. The fused scheduler excludes
  misplaced experts from both tiers. The load plan now carries every rank's
  final placement and derives the global hot mask from those plans. New CPU
  regressions fail before the fix for both directions; all 47 manifest tests
  pass afterward. Patch: `source-residency-fix.patch`.
- The route-check flag previously accumulated local fingerprints without any
  cross-rank comparison. Qualification now compares the full route-ID tensors
  with an asynchronous assertion, avoiding both missing checks and hash
  collisions. Two CPU diagnostic tests pass; graph-mode validation is pending.
- Job 1671834: 14 GPU replica tests passed; server startup then exposed missing
  vendored Python files in the detached checkout. Copied missing runtime files
  and resubmitted as 1674081. Earlier 1671811 failed because compute nodes lack
  git; 1671819 was cancelled before testing to fix the residency-mask bug.
- Job 1674081: same-node c1/MTP3 four-arm screen, two order-balanced rounds,
  12 requests x 256 fixed output tokens per arm. Timing is streamed and saves
  TTFT/TPOT, text chunks, usage and Prometheus snapshots.
- Actual daily production is DFlash2/c1/DCP1, confirmed by recent Slurm job
  history and the c1-df2 launcher. Its plan retains 2,224 hot experts per rank,
  versus 2,325 for the MTP3 trace configuration. Job 1674124 qualifies this
  shape independently, using `dflash2-joint/candidate.json`, route-checked smoke,
  three order-balanced latency pairs, 96K-context requests and full paired GSM8K.

### Corrected GPU screen and DFlash2 startup investigation

- Found and fixed a third correctness defect: the static cold expert map
  included secondary copies even when prefill bypassed replica assignment.
  Static execution now excludes secondaries while the physical map retains
  them for the decode scheduler. Its CPU regression fails before the fix.
  Job 1674081 was cancelled and its timings are **not qualification evidence**.
- Job **1674389** completed all eight MTP3 arms with all three fixes. Two
  same-node order-balanced rounds, 12 short requests x 256 output tokens:

  | MTP3 c1 arm | TPOT (ms) | Decode time / draft round (ms) | Acceptance length |
  | --- | ---: | ---: | ---: |
  | Current / off | 11.254 | 29.114 | 2.595 |
  | Current / exact | 10.472 | 27.346 | 2.619 |
  | Joint / off | 11.455 | 29.920 | 2.627 |
  | Joint / exact | 10.889 | 26.965 | 2.483 |

  Exact assignment reduced time per draft round by 6.1% with the current
  placement and 7.4% with the joint candidate, relative to current/off.
  This metric is server decode duration divided by drafts at c1, not a
  hardware-trace engine span. The new placement alone did not improve this
  workload; its offline proxy is insufficient to justify deployment.
  Acceptance shifts also make TPOT differ from the round-time result.
  Two pairs are an initial screen, below the rollout gate. Raw data and
  paired confidence intervals: `results-1674389/summary.json`.
- DFlash2 jobs **1674124** and **1674391** failed during initial model storage
  allocation, before checkpoint streaming or route-checked inference.
  The latter passed all **23 GPU/CPU replica tests** first. A diagnostic in
  the frozen qualification checkout identifies rank 3 exiting with SIGKILL
  (-9); this alone does not establish OOM. The diagnostic is not in root
  production source. Full graph-mode validation remains pending.
- Job **1683608** records per-NUMA free memory, process RSS, and cgroup
  memory limits/events while trying candidate/exact and current/off with
  production prefetch enabled. This is a startup diagnostic, not a latency
  comparison. `memory_watch.py` records the raw observations.
- The optimizer now accepts `--replica-budget` and `--freeze-hot`. Smaller
  budgets seed secondaries by training-only distinct activation counts,
  then optimize the same exact cold-load objective. A 512-copy seed with
  DFlash2/prefetch retains 2,166 hot experts per rank and budgets a 1,175 MiB
  staging slot, versus 2,182 hot and 851 MiB with current/off. Candidate
  staging capacity must be replanned after optimization before benchmarking.

### Pinned allocator accounting and combined prefetch candidates

Job **1683608** confirms OOM: cgroup `memory.events` increments `oom_kill`
from 0 to 4 during candidate construction; sampled worker RSS reaches
108.6 GiB. PyTorch's installed `ATen/core/CachingHostAllocator.h` rounds each
pinned allocation to a power of two, while the planner previously counted
only logical weights. With prefetch off, the original DFlash2 joint profile's
70.4 GiB logical cold storage needs 104.75–109.25 GiB of allocator backing
per rank. Prefetch demotions worsen the problem. Baseline/off passed storage
construction and weight loading in this diagnostic, then failed because the
diagnostic launcher omitted the explicit Triton cache path. Both statuses
are recorded as not ready; the Slurm exit code alone is not a pass.

The root planner now includes per-layer pinned rounding in host totals and
rejects overflow before allocation. Both new regressions fail before the
fix; all **49 manifest/planner tests pass** after it. Patch:
`source-memory-accounting.patch`. The running qualification checkout remains
frozen at the preceding three correctness fixes; this accounting-only patch
does not change model execution. It must be included in final rollout checks.

The same allocator threshold informs a new joint-search constraint:
`--max-cold-experts-per-layer 50`. Fifty group-32 experts fit below 1 GiB;
51 cross into a 2 GiB pinned block. Search constrains residency demotions and
replica destinations together, so more replicas can fit with less backing.
No held-out request is used to choose moves.

| Combined DFlash2 / prefetch candidate | Hot/rank | Replicas/rank | Pinned cold GiB/rank | Staging MiB |
| --- | ---: | ---: | ---: | ---: |
| Current / off | 2182 | 0 | 71.5–72.0 | 851 |
| Joint / 512 | 2170 | 512 | 73.25–77.75 | 1094 |
| Joint / cap 50 | 2174 | 919–934 | 73.5–74.0 | 1013 |

These candidates are frozen as `df2-prefetch-512-joint/runtime.json` and
`df2-prefetch-cap50-joint/runtime.json`, after replanning the optimized
staging slot. Their offline metrics remain count proxies.

- **1683673**: 512-copy screen, graph-mode route-check smoke followed by
  current/off, current/512-exact and joint/512-exact. Three rounds rotate
  order so each arm occupies each position once; short and 96K-context timings.
- **1683773**: cap-50 qualification, graph-mode route-check smoke, three
  baseline/off versus candidate/exact timing pairs with prefetch enabled,
  and full paired 1,319-question GSM8K with a route-checked candidate soak.

Neither job's submission constitutes qualification. Production assignment
remains off pending their results and the final trace comparison.

### Padding invariant caught by the graph-mode check

The 512-copy candidate in **1683673** passed storage construction, checkpoint
loading and compiled warmup, then the route-ID comparison asserted during
PIECEWISE graph capture. The V2 capture path explicitly marks every dummy
row as padding; its uninitialized router values can differ across ranks.
Replica assignment previously consumed those padding IDs as real work.
This is also relevant to partially padded real batches: unused rows must
not influence the shared assignment decision.

The fix masks padded rows to -1 **before both assignment and its route
comparison**, using the runner's existing device padding mask. The fused
histogram and alignment now skip invalid IDs, so they do not scatter at a
negative index. The full cross-rank check remains present in captured graphs
and checks every nonpadding route. CPU coverage verifies the mask changing
from all-dummy to a real batch; GPU coverage checks that all-dummy and
partially padded inputs schedule exactly the real routes.

Jobs 1683673, 1683773 and the newly submitted trace job 1683796 were stopped;
none produced qualification results. The frozen checkout now includes the
padding fix and pinned-allocation accounting, with updated hashes in
`qualification-provenance.json`. Restarts are **1684062** (cap-50 full
qualification) and **1684063** (512-copy three-arm screen), each running
the complete 26-case replica suite on its GPU allocation before serving.
Trace capture will restart after this graph-mode gate succeeds.

The final physical profiles reproduce their planned hot sets exactly.
`final-runtime-offline-comparison.json` reports the held-out eight-token
proxy at those placements: current/off 434.47, current/512-exact 370.50,
joint/512-exact 290.38, joint/cap-50-exact 274.11. These remain offline
expert-count units, not observed latency improvements.

## Frozen offline candidates

Owners are unchanged. Search uses distinct active experts, exact Hall subset
bounds, per-rank hot swaps and replica relocation/addition, with at most 985
copies per destination rank. The formula was checked against the previous
max-flow solver on 200 independent random layer problems. Every training
request contributes equally; the request holdout is not used in search.

| Held-out c1/MTP3 proxy | Sum of layer maxima | Active cold experts |
| --- | ---: | ---: |
| Current placement / off | 260.96 | 610.57 |
| Current placement / exact | 206.60 | 610.57 |
| Joint candidate / off | 209.84 | 461.19 |
| Joint candidate / exact | 162.58 | 461.19 |

These are expert-count units, not milliseconds. Exact rows model corrected
runtime masks; the old buggy exact path is not a valid numerical baseline.
The DFlash2 candidate uses adjacent four-token recorded MTP verification groups
as an eight-token proxy, never crossing request boundaries. That is not a
capture of actual DFlash2 routing; served-model validation must establish
whether it transfers. Its held-out proxy is 422.77 current/off, 338.41
current/exact and 261.53 joint/exact. Reports preserve per-request scores and
alternative assumed compute floors.

## Predeclared rollout criteria

- Zero transport errors, exactly-once routing tests and a successful full
  graph-mode cross-rank route-check soak on the production shape.
- Paired GSM8K, all 1,319 questions: report the accuracy difference and paired
  request-bootstrap 95% confidence interval. Noninferiority margin is 1.0
  percentage point; its lower bound must exceed -1.0 point. A nonsignificant
  McNemar p-value alone does not pass this gate.
- At least three same-node order-balanced latency pairs; a repeatable TPOT
  improvement with acceptance separately reported, no unexplained TTFT or
  long-context regression. Profile the winning comparison to confirm the
  predicted reduction in cold-work tails and collective waiting.
- Check the final production prefetch setting and replan/measure the combined
  configuration before rollout. Replicas enlarge the staging slot: the MTP3
  candidate needs 1,276 MiB with prefetch on and retains 2,262 hot experts,
  rather than the unstaged 2,325. Existing prefetch timing cannot be assumed
  unchanged when replicas become materialized.

### Attributing the smoke-gate failure (job 1685116)

Restarts **1684062** and **1684063** both died at the route-checked smoke with
accuracy 0.0 (`preds [-9999999, 2, ...]` against labels `[18, 3, ...]`), so
neither produced qualification evidence. The servers themselves were healthy:
PIECEWISE, FULL and dflash2 captures all completed and the cross-rank route
check never asserted. All four ranks therefore agree on their routes, which
rules out the cross-rank divergence class and points at a defect that alters
routing identically on every rank.

The smoke cannot attribute its own failure: it only ever runs
candidate/`exact`/`ROUTE_CHECK=1`, with no baseline arm. Job **1685116** runs a
four-rung ladder that changes one variable per rung, 8 questions each, on the
same frozen checkout and with production prefetch at 1,024 tokens:

| Arm | Profile | Assignment | `ROUTE_CHECK` |
| --- | --- | --- | --- |
| A | baseline | off | 0 |
| B | baseline | off | 1 |
| C | baseline | exact | 1 |
| D | cap-50 candidate | exact | 1 |

The first rung that collapses names the cause: A implicates the frozen checkout
itself (arm A is the production DFlash2 configuration), B the route check, C the
assignment path, D the candidate placement. `diagnose_smoke.py` always writes
its result and keeps the completions, unlike the qualification gate, which
refuses a zero by design and so cannot be used to compare arms.

**Result (job 1685116, 8 questions per arm):**

| Arm | Profile | Assignment | `ROUTE_CHECK` | Accuracy |
| --- | --- | --- | --- | ---: |
| A | baseline | off | 0 | 0.875 |
| B | baseline | off | 1 | 0.875 |
| C | baseline | exact | 1 | server died, rank 3 exitcode -9 |
| D | cap-50 candidate | exact | 1 | **0.000**, invalid 0.375 |

A and B are byte-identical in their predictions, so the frozen checkout is
healthy and the route check is both working and free of side effects. C hit
the already-known pinned-rounding OOM, because the baseline profile's 3,940
unconstrained secondaries exceed host memory once materialized and the
accounting patch is deliberately not in this checkout. D reproduces the
failure, and its garbage differs on every run (`[-9999999, 0, ...]` here,
`[-9999999, 2, ...]` in 1684062 and 1684063) while A and B are deterministic.

### Root cause: `is_padding` is published unconditionally but maintained conditionally

`prepare_inputs` refreshes the padding buffer only under `VLLM_MOE_SKIP_PADDING`
(`gpu/model_runner.py:871-878`), which defaults to false and is set by no serve
script. Graph capture fills the same buffer with all-`True` unconditionally
(`gpu/cudagraph_utils.py:496`), and the forward context publishes it
unconditionally (`gpu/model_runner.py:1038`). So after capture the buffer is
stuck all-`True` for the life of the server.

`mask_replica_padding` reads it without that guard, so every real token's
`topk_ids` becomes -1; the assignment kernel's new `routed >= 0` test then
masks out every route and no expert executes, leaving whatever was already in
the output rows. That accounts for all four observations: nondeterministic
garbage, identical on every rank so the route check stays silent, only under
`exact` because `apply_tiered_moe` gates the mask on the assignment mode, and
invisible to arms A and B, which never call it.

The contract is settled by the pre-existing consumer, which guards correctly:
`deepseek_v4/nvidia/model.py:455` reads `is_padding` only under
`if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available()`. The
qualification patch introduced a second consumer that omits the guard. This is
a live latent defect in root production source, currently masked only because
production ships replica assignment off.


### Fix and validation (job 1763631)

`prepare_inputs` now refreshes the padding mark unconditionally, through a
named `_mark_padding`, so the buffer matches the contract its unconditional
publication already implies. Root-tree commit `8b131c5650`, with a regression
in `tests/v1/worker/test_gpu_model_runner_v2_eplb.py` that fails on the gated
version and passes on the fixed one; all 75 tiered replica/manifest tests pass.

The originally planned second half -- confining the mask to assignment and
validation instead of rebinding `topk_ids` -- was dropped after reading
`apply_tiered`. The rebinding is the upstream design: `-1` rows are meant to
flow into `_prepare` so prepare/finalize drops them, which is the same
mechanism `modular_kernel.py:1250` describes. Masking only the assignment
would make genuine padding rows execute.

What reading could not settle is whether the `-1` sentinel survives the
tiered Marlin path end to end. The assignment kernel is guarded (`routed >= 0`),
but arm D's smoke cannot exercise the sentinel at all: with capture size 8 and
MTP K=7 every decode step is exactly at the capture boundary, so `is_padding`
is all-False and no `-1` is ever produced. Job 1763631 therefore runs three
arms, the second of which replays a size-16 graph with 8 real rows:

| Arm | Profile | Assignment | Capture | Purpose |
| --- | --- | --- | --- | --- |
| 1 | cap-50 | exact | 8 | reproduces 1685116 arm D, which scored 0.000 |
| 2 | cap-50 | exact | 16 | 8 padding rows carry `-1` into the tiered path |
| 3 | cap-50 | off | 16 | control, so an arm 2 failure is not the capture size |

Arms 1 and 3 passing with arm 2 failing would mean the fix is correct and the
`-1` sentinel is separately unsupported by this Marlin path, not that the fix
failed.

**Result (job 1763631).** All three arms score 0.875 with byte-identical
predictions `[18, 3, 70000, 540, 20, 64]`, matching arms A and B of job
1685116 exactly.

| Arm | Assignment | Capture | Before | After |
| --- | --- | --- | ---: | ---: |
| 1 boundary | exact | 8 | 0.000 | **0.875** |
| 2 padded | exact | 16 | - | **0.875** |
| 3 padded | off | 16 | - | **0.875** |

Arm 1 is the configuration that scored 0.000 in 1684062, 1684063 and 1685116,
so the fix is confirmed against its own failure. Arm 2 settles the question
reading could not: the server logs record `capture_sizes': [8]` for arm 1 and
`[16]` for arm 2, so arm 2 really did replay a size-16 graph with 8 real rows
and push 8 sentinel-carrying padding rows through the tiered Marlin path. The
sentinel is tolerated end to end, and arm 3 shows the padded shape is not
itself responsible for anything.

Identical predictions across assignment on and off is the expected signature
of a correct implementation: replicas move where an expert is read from, not
what it computes.
