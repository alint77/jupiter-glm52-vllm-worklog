# Review: replicated Grace experts with per-step makespan scheduling

Reviewer: Claude Opus 4.6, 2026-07-31. Scope: `README.md` in this directory, the
raw result JSONs here, and vLLM commit `1a94ed458` ("Add replica-aware tiered MoE
scheduling", 18 files, +1446/-42).

**Verdict: the mechanism is real and the engineering is the most disciplined in
this worklog. The headline performance numbers are the weakest part of it, and
one named safety mitigation was designed but not implemented.** Nothing here
needs to be thrown away. Five things need fixing, in the order below.

Everything in this review is reproducible from artifacts already in the repo;
commands are given inline.

---

## Part 1 — What is solid and should not be touched

Listing this explicitly so the follow-up work does not "fix" things that are
already right.

1. **The problem selection is evidence-driven.** The load-bearing insight is in
   Phase 0: replaying captured routes showed only ~14% of modelled skew is
   removable by a better *static* owner map, ~86% is the per-step
   max-of-four-ranks effect. That is the correct argument for replicas, and it is
   the reason this is a better use of time than another placement-optimizer pass.

2. **Gate discipline.** Offline oracle before any runtime change, memory probe
   before the loader, correctness before performance, with numeric go/no-go
   thresholds set *before* the measurements. Keep this structure.

3. **The negative controls are real and were reported.** Hash selection predicted
   16.03/45.34 ms and token-count least-loaded 14.74/40.38 ms against a
   14.14/36.13 ms baseline - both worse. The static `secondary` arm measured
   -11.5%/-13.5%. An experiment that only reports the policy that worked is much
   less trustworthy than this one.

4. **The census invariance check is exactly the right test for this mechanism.**
   300 routed Marlin launches, 157 custom all-reduces, 177 NCCL all-gathers per
   rank-step, identical in both arms. That establishes the win came from *which
   copy executes*, not from altering communication.

5. **Withdrawing the exact-text semantic gate was correct.** Discovering that
   off-vs-off diverges at a median of 3.5 tokens across restarts, and therefore
   refusing to claim a pass from 0/8 exact matches, is the right call. A weaker
   report would have shipped that as evidence of safety.

6. **The offline model validated against reality.** The oracle predicted
   1.74 ms (c1) and 3.95 ms (c4) net savings; measured step-time savings are
   1.245 ms and 5.079 ms. Same order in both regimes, which is a genuine
   endorsement of the replay methodology.

7. **The 1,697-copy runtime failure was handled correctly.** Job `1122219` lost a
   worker despite the isolated probe passing at that footprint; falling back to
   985 copies and saying so is right. See item 5.4 for the caveat this creates.

---

## Part 2 — Fix in this order

### 2.1 (Correctness) The exactly-once invariant rests on an unenforced precondition

**Severity: highest. A violation is silent — wrong tokens, no error.**

The design document requires the feature be *"enabled only for an execution
layout whose rank route IDs have been validated as identical"*, and the risk
table specifies *"restrict the feature to validated layouts; fail closed rather
than add a dispatch collective."* That mitigation is not implemented.

`vllm/config/tiered_moe.py:53-63` validates only:

```python
if self.replica_assignment != "off" and not self.enabled: ...
if self.replica_assignment != "off" and self.placement_profile is None: ...
```

There is no check on TP == EP, on custom all-reduce being enabled, on DP
attention, or any runtime verification.

**Why it currently works.** I verified the underlying assumption holds today.
`csrc/custom_all_reduce.cuh`, in `cross_device_reduce_1stage`:

```cpp
// note: we don't reorder the address so the accumulation order is the same
// for all ranks, ensuring bitwise identical results
```

Identical hidden states across ranks → identical router logits → identical
top-k → identical `topk_ids` → identical greedy assignment. The two-stage variant
is also cross-rank identical (each element reduced once, then gathered).

**Why that is not enough.** `--disable-custom-all-reduce` falls back to NCCL
ring all-reduce, which carries no cross-rank bitwise guarantee. If one rank's
top-k differs by a single expert, that expert is executed by **zero** ranks
(output silently dropped) or **two** (silently double-counted by the existing
all-reduce). Nothing asserts, nothing logs.

**The existing test cannot catch this.**
`tests/model_executor/model_loader/test_tiered_moe_manifest.py:618`
(`test_gpu_greedy_replica_assignment_executes_every_route_once`) passes the *same*
`routes` tensor to both simulated ranks. It proves the algorithm is deterministic
given identical inputs. The risk is that the inputs are not identical.

**Action:**

- In `validate_tiered_moe_config`, fail closed when `replica_assignment != "off"`
  and any of: custom all-reduce disabled, `tensor_parallel_size != ep_size`, or a
  data-parallel/sharded-routing layout is configured. Error message should name
  the reason ("replica assignment requires bitwise-identical cross-rank routing").
- Add an opt-in debug check (env-gated, off by default) that all-reduces a hash
  of `topk_ids` once per N steps and raises on mismatch. This is the only way to
  detect divergence if a future layout breaks the assumption.
- Add a test that constructs *divergent* per-rank routes and asserts the guard
  fires, rather than only testing the identical-input path.

### 2.2 (Measurement) Every A/B pair ran on a different node, at n=2

```
c1:     off jpbo-006-48    greedy jpbo-116-34
c4:     off jpbo-004-39    greedy jpbo-005-04
trace:  off jpbo-032-37    greedy jpbo-027-29
```

Reproduce: `sacct -j 1123052,1123054,1123095,1123096,1123358,1123364 --format=JobID,NodeList,State,Elapsed`

This project has already measured what cross-node comparison costs. From
[`2026-07-29-marlin-smem-monopoly`](../2026-07-29-marlin-smem-monopoly/README.md):
*"The earlier cross-node +9.71% was inflated by about 2.5 points of node and
acceptance luck; ~+7% is the honest figure."* `job-samenode.sh` exists in that
directory for exactly this and was not used here.

Welch's t on output throughput, from the raw per-repeat values in
`phase4-runtime-summary.json`:

| regime | off | greedy | Δ | t | df |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1 | 91.89, 93.08 | 95.64, 95.83 | +3.51% | 5.37 | 1.05 |
| c4 | 149.42, 146.86 | 160.70, 155.54 | +6.74% | 3.46 | 1.46 |

Neither resolves at n=2, and that is *before* the node confound, which
within-arm variance cannot see at all.

Note also that the c4 noise is not cleanly acceptance-driven: greedy's repeats
move *with* acceptance (58.78%→160.70, 56.23%→155.54) while off's move *against*
it (56.07%→149.42, 57.62%→146.86). Whatever is driving c4 variance is not a
single scalar.

**Action:** one allocation, both arms sequentially on the same node, following
`../2026-07-29-marlin-smem-monopoly/job-samenode.sh`. Three repeats per arm.
This is the single measurement that converts the headline from "p≈0.07,
cross-node" into something defensible.

### 2.3 (Measurement) The acceptance correction is missing, and it strengthens your result

`step_time = TPOT × accepted_tokens_per_step` divides draft acceptance out of the
comparison analytically. It costs nothing — the inputs are already in
`phase4-runtime-summary.json`. Computed:

| regime | throughput | TPOT | accept length | **acceptance-corrected step time** |
| --- | ---: | ---: | ---: | ---: |
| c1 | +3.51% | -3.67% | -0.89% | **-4.52%** |
| c4 | +6.74% | -8.67% | +0.73% | **-8.01%** |

Two consequences the report misses:

- **At c1, greedy wins despite lower acceptance.** That is a stronger claim than
  +3.51%, and it moves c1 from "below the 5% success threshold" to -4.52% step
  time. Combined with 2.2, **c1 is the better-supported arm** (t=5.37 against
  3.46; its variance is 6× smaller). The README leads with c4 and semi-dismisses
  c1; the statistics support the opposite ordering of confidence.
- At c4 the correction slightly *reduces* the claim, from -8.67% TPOT to -8.01%
  step time.

**Action:** report acceptance-corrected step time as the primary metric for every
kernel-level arm from now on, alongside raw throughput. This is the metric the
shared-memory phase used to get its "step time falls 7.66%" figure.

### 2.4 (Reporting) The trace section quotes the flat metric and omits the one that agrees

`phase4-trace-analysis.json` contains three cycle metrics. The README quotes two:

| metric | off | greedy | Δ | quoted in README? |
| --- | ---: | ---: | ---: | --- |
| `graph_start_cycle_ms` | 51.993 | 51.463 | -1.02% | yes |
| `graph_span_ms_per_rank_step` | 49.329 | 48.932 | -0.81% | yes |
| **`annotation_start_cycle_ms`** | **47.196** | **44.998** | **-4.66%** | **no** |

The omitted one is the `execute_` annotation cycle — the engine-step definition
this worklog standardised on in the shared-memory phase — and it is the one
directionally consistent with the benchmark's -8.01%. Note also that the two
"off" cycle metrics disagree by 10% *within the same arm* (47.196 against
51.993), on 24 rank-steps drawn from 6 graphs per rank.

The supportable conclusion is not "the graph cycle is nearly flat, so the
profiled capture disagrees with the benchmark." It is:

> The trace is **decisive on the mechanism** — summed layer max-minus-mean
> -44.2%, custom all-reduce residency -44.9%, kernel census unchanged — and
> **underpowered on step time**, with its own cycle metrics differing by 10%
> within an arm.

**Action:** rewrite the "Paired c4 trace result" narrative to state both, quote
all three cycle metrics, and stop using the flat one as evidence that the
end-to-end gain is suspect. If a step-time verdict is wanted from a trace,
capture more than 6 graphs per rank.

### 2.5 (Physics) The counter-cost is 94% of the gross win, and two constants govern it

Per rank-step from the paired c4 trace:

| | ms |
| --- | ---: |
| custom all-reduce residency | **-4.388** |
| routed Marlin cumulative | **+3.972** |
| replica scheduler | **+0.699** |
| net (cumulative) | **+0.283** |

The saving in all-reduce wait is almost exactly consumed by moving work from HBM
primaries to Grace secondaries. That trade is controlled entirely by
`vllm/model_executor/model_loader/tiered_moe_scheduler.py:12-13`:

```python
_HBM_COST = 1280
_GRACE_COST = 3467      # ratio 2.709
```

The ratio is consistent with the c1/M=4 measurement in the shared-memory phase
(cold 3.980 ms for ~20% of routed traffic against hot 5.917 ms for ~80% gives
2.69 per expert task). Two problems with applying it unchanged:

- **It is a c1/M=4 calibration used at c4/M=16.** At M=16 the hot tier does more
  compute per expert while cold stays weight-bandwidth-bound, so the true ratio
  should be *lower* than 2.71 — the scheduler is working from a stale constant in
  precisely the regime where it claims its win.
- **Cost is flat per expert task, ignoring token count.** The design document
  required cost to depend on "token count for that expert where it materially
  changes cost"; the implementation uses `counts` only for ordering
  (`tiered_moe_scheduler.py:199-203`). At M=16 the per-expert token spread is
  much wider than at M=4, so this approximation costs more there.

**Action, and this outranks the fusion experiment:** re-measure the hot/cold
per-expert-task cost at M=16 using `benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py`,
then either update the constants or make them shape-dependent. Add token count to
the incremental cost if the re-measurement shows it matters. **Expected value is
higher than removing the 0.699 ms scheduler**, because the counter-cost is
3.972 ms.

---

## Part 3 — Implementation issues

### 3.1 Unbounded Triton specialisation on route count

`tiered_moe_scheduler.py:103` declares `NUM_ROUTES: tl.constexpr`, and
`:332` passes `NUM_ROUTES=topk_ids.numel()`. `:341` also switches `num_warps`
at 32. The op is invoked on **every** forward, including prefill — see
`tiered_moe_execution.py:314`, where `schedule=` merely selects the early-return
branch inside the kernel; the constexpr still specialises.

Every distinct prefill token count therefore triggers a fresh Triton compile, and
nothing buckets or caps it. `NUM_ROUTES` does not need to be `constexpr`: it is a
loop bound (`:159`) and a priority scale (`:199`). Symptom is JIT stalls on new
shapes, not a wrong answer.

**Action:** make it a runtime argument, or skip the launch entirely when
`schedule=False` and restore the primary maps once at capture time.

### 3.2 Scheduler cost is 2.8× its own budget, on a single CTA

Measured 0.699 ms at c4 (9.31 us × 75 layers) against the stated 0.25 ms gate;
0.297 ms at c1 also misses. The kernel launches with grid `(1,)` — the entire
scheduler is one CTA on one SM, running a sequential `while` loop over up to 128
flexible experts (`:213-214`). Latency-bound by construction.

The fusion experiment targets this correctly. Note its own honest bound of 1-2%,
and see 2.5 for why it should not go first.

### 3.3 `selected_ranks` is reused as undocumented scratch

`tiered_moe_scheduler.py:203-214` sorts the expert order, stores it into
`selected_ranks_ptr`, issues `tl.debug_barrier()`, then reads it back
element-by-element inside the loop, before finally overwriting the buffer with
the real selected ranks at `:298`.

This works — Triton cannot index registers dynamically, so the global round trip
is the standard workaround, and the barrier makes it visible within the single
CTA. But the buffer is also a declared `mutates_args` output, so for the duration
of the loop a declared output holds unrelated data. It is correct and
undocumented, which is the worst combination for the next person.

**Action:** add a comment explaining the reuse and the barrier's role, or use a
separate scratch buffer.

### 3.4 Smaller

- **The deployed profile is 985 copies, not the 1,697 the oracle selected.** Job
  `1122219` lost a worker at the 1,697 footprint despite the isolated probe
  passing. Reported honestly, but it means the oracle's headline sweep is not the
  deployed configuration, and the Phase 1 table should say so at the point where
  1,697 is bolded.
- **HBM headroom is 3,874 MiB at c4 greedy.** Flagged in the README. Worth a
  hard preflight check rather than an observation.
- **Run-to-run non-determinism was found but never root-caused.** It is most
  likely batch-composition-dependent kernel selection, but it is also the exact
  signal that would reveal cross-rank routing divergence (2.1). Ten minutes with
  a fixed-batch replay would separate the two.

---

## Part 4 — Measurement protocol for follow-up work

The question came up of whether to replace MTP3 runs with a "simulated MTP3"
control: no speculative decoding, but at concurrency scaled so the target model
sees the same M (no-MTP c4 for MTP3 c1, no-MTP c16 for MTP3 c4). This is a good
idea with one quantified caveat, and it corrects a gap in the existing protocol.

**The existing no-MTP protocol measures the wrong shape.**
`../2026-07-29-marlin-smem-monopoly/job-nomtp.sh` used concurrency 1/2/4, giving
M = 1, 2, 4. Production MTP3 is M=4 at c1 and **M=16 at c4**. Nothing has ever
been measured at M=16 without MTP — the shape where this experiment claims its
main result.

**The substitution is biased on the variable under test.** Measured on the 22
held-out requests of the 108-request Claude Code routing capture, via
`measure_route_regime.py` (added by this review; results in `route-regime.json`):

| shape | construction | distinct experts/layer | per rank |
| --- | --- | ---: | ---: |
| M=4 | **MTP3 c1**: 4 consecutive tokens, 1 seq | 25.27 ± 1.71 | 6.32 |
| M=4 | no-MTP c4: 4 independent tokens | 29.71 ± 0.66 | 7.43 |
| M=16 | **MTP3 c4**: 4 seq × 4 consecutive | 81.71 ± 3.97 | 20.43 |
| M=16 | no-MTP c16: 16 independent tokens | 93.03 ± 2.10 | 23.26 |

Independent sequences inflate distinct activated experts by **+17.6% at M=4 and
+13.9% at M=16**. Adjacent tokens genuinely route to overlapping experts, and MTP
verification positions are adjacent tokens.

That is not a nuisance variable for this experiment — it *is* the objective. The
greedy scheduler balances per-rank makespan over distinct expert tasks, so 20.43
against 23.26 per rank is a 14% change in the load being balanced, and the extra
experts come from the routing tail, which is disproportionately cold. Two further
differences: 16 independent sequences carry 4× the KV footprint of 4 sequences
(relevant at 3,874 MiB free), and no-MTP omits the draft passes, so the
denominator differs and percentages are not directly comparable to production.

**Recommended protocol, cheapest first:**

1. **Acceptance-corrected step time on production MTP3 runs.** Free, exact
   shapes, exact route correlation. Use as the primary metric — see 2.3.
2. **Reject-all MTP3, if you are willing to patch.** Run MTP3 but accept only
   draft position 0. Under greedy sampling position 0 is the target's own token
   and is always accepted, so emitted text is deterministic and identical to the
   no-MTP run, while the target forward still runs at M=4C with the correct
   verification-position correlation. `step = TPOT` exactly, no correction
   needed. Throughput drops to roughly a third of production, which does not
   matter when the measurement is step time. Roughly 20 lines behind an env flag.
3. **No-MTP at matched M** (concurrency 4 and 16, not 1/2/4). Free and fully
   deterministic to ±0.1 ms. Good as a confirmatory arm and for levers that do
   *not* scale with distinct-expert count. For tiered-MoE work, read it knowing
   it overstates the cold-tier share by 14-18%, and never cross-compare its M=4
   point directly against MTP3 c1's M=4.

Reproduce the table:

```bash
.venv/bin/python agent_space/experiments/2026-07-31-replicated-expert-scheduling/measure_route_regime.py \
  --json route-regime.json
```

---

## Part 5 — Recommended order of work

1. **Config fail-closed guard** (2.1). Twenty lines. Turns "safe by luck" into
   "safe by construction".
2. **Same-node A/B with acceptance-corrected step time** (2.2, 2.3). One
   allocation. Converts the headline into a defensible number.
3. **Recalibrate `_HBM_COST`/`_GRACE_COST` at M=16, add token-count sensitivity**
   (2.5). This is where the remaining performance is: the counter-cost is 3.972 ms
   against the scheduler's 0.699 ms.
4. **Fix `NUM_ROUTES` specialisation** (3.1). Small, prevents a latency-spike
   class of bug.
5. **Rewrite the trace narrative** (2.4) and the Phase 1 deployed-profile caveat
   (3.4).
6. **Then** the fusion experiment in
   [`2026-07-31-replica-align-fusion`](../2026-07-31-replica-align-fusion/README.md).
   It is correctly scoped and honest about its 1-2% bound, but it optimises the
   0.699 ms scheduler while the 3.972 ms counter-cost sits untouched.

---

# Round 2 — review of the response (worklog `ad498a9`, vLLM `dc4dcff58`)

All five items were addressed, one of them by correctly refuting me. The
hardening is good. The measurement work is much better than round 1 and now
supports a **different and cleaner headline than the one written**. One new
reproducibility problem was introduced that undermines the cost-policy decision.

## 2.0 Closed correctly

- **2.1 fail-closed guard** — `validate_replica_routing_layout` in
  `vllm/config/tiered_moe.py` rejects disabled custom all-reduce, `TP != EP`,
  `DP != 1`, `PP != 1`, and disabled expert parallelism, with the reason named in
  the error. Good coverage. (Minor: `ep_size = tp * dp` makes the `tp != ep_size`
  and `dp != 1` checks redundant with each other.)
- **2.1 runtime check** — `validate_replicated_routes` hashes routes, all-reduces
  the hash, and raises on divergence. See 2.9 for its limits.
- **3.1 Triton specialisation** — `NUM_ROUTES` is now a runtime argument.
  (`num_warps` still switches at 32, which is two bounded variants — fine.)
- **3.3 scratch reuse** — documented at the point of use.
- **2.4 trace narrative** — rewritten, `execute_` annotation cycle at -4.7% now
  in the table, "decisive on mechanism and underpowered on step time" stated.
- **3.4 deployed-profile caveat** — added at the Phase 1 table.
- **HBM preflight — my item was wrong.** `tiered_moe_physical.py:97-126` already
  reads physical free HBM after warmup and fails closed below the reserve with a
  5 GB floor. The pushback is correct; disregard that part of round 1.

## 2.6 The status line takes the best metric from each regime

> "-6.75% corrected target-step time at c1 and +6.31% aggregate throughput at c4"

Those are the two favourable cells. The other two are **c1 throughput -1.94%**
and **c4 corrected step -3.42%**. Since the report itself argues corrected step
should be primary for scheduler changes, mixing metrics between regimes in the
status line undercuts that argument.

## 2.7 The statistics support a better headline than the one written

Welch's t on the three same-node repeats of the retained policy
(`same-node-legacy-summary.json`, jobs `1133046`/`1133047`):

| metric | off | greedy | Δ | t | df | reading |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c1 throughput | 94.357 | 92.526 | -1.94% | -1.28 | 2.79 | not significant |
| **c1 corrected step** | 27.921 | 26.035 | **-6.75%** | **-9.73** | 2.74 | **strong** |
| c1 accept length | 2.930 | 2.683 | -8.45% | -8.69 | 2.90 | strong, adverse |
| **c4 throughput** | 150.222 | 159.706 | **+6.31%** | **11.87** | 2.74 | **strong** |
| c4 corrected step | 62.127 | 60.003 | -3.42% | -2.84 | 3.23 | marginal |
| c4 accept length | 2.718 | 2.804 | +3.19% | 1.87 | 3.07 | marginal |

Read consistently, on the metric the report itself nominates:

> **Corrected target-step time falls 6.75% at c1 (t=9.7) and 3.42% at c4
> (t=2.8). Aggregate throughput moves ±3% around that depending on acceptance
> luck — unlucky at c1 (-8.45% accepted/step), lucky at c4 (+3.19%).**

That is a stronger and more defensible claim than the current status line,
because it holds in both regimes and does not depend on which metric is chosen.
Note the reversal from round 1: cross-node made c4 look like the strong arm; the
same-node data makes **c1** the strong arm on step time and c4 marginal.

**Action:** replace the status line with the corrected-step figures for both
regimes and report throughput as a secondary, acceptance-confounded number.

## 2.8 The acceptance shift needs explaining, not just noting

Paired greedy-minus-off acceptance-length deltas across the three same-node
experiments:

| policy | Δ accepted/step |
| --- | ---: |
| original 2.709 | **-8.45%** |
| chain | -2.75% |
| interim scalar | **+2.63%** |

Within each pair the repeat SDs are 0.02-0.09, so each delta is individually
"significant". Across pairs the **sign flips**. That is the signature of chaotic
sensitivity, not a systematic effect: replica assignment changes which rank
reduces which expert, so the fp32 accumulation order changes, greedy text
diverges within a few tokens, and acceptance lands wherever it lands.

This matters in two directions and the README states neither:

- **Reassurance.** A -8.45% acceptance drop looks like a model-quality
  regression. It is not — the sign is not reproducible. Say so explicitly, or a
  future reader will treat it as evidence the scheduler harms drafting.
- **Consequence.** Acceptance deltas between arms carry no information, so
  aggregate throughput cannot be compared between arms at n=3 in either
  direction. This is the concrete justification for the corrected-step metric,
  and it is stronger than the current "acceptance varies substantially".

## 2.9 The route check cannot run in the configuration that ships

Two limits, both worth recording next to the feature:

- **It requires `--enforce-eager`** (`vllm/config/vllm.py`), while production runs
  full CUDA graphs. Kernel selection and reduction order differ between eager and
  graph replay, so "the check passed" validates a different execution path than
  the one deployed. It is still worth having; it is not proof about production.
- **It covers only the first routed layer** — `tiered_moe_physical.py:168` sets
  `tiered_replica_route_check = layer_offset == 0`. Divergence originating at a
  deeper layer is caught only indirectly, on a later token, once it has
  propagated back to layer 0's input.

Minor: `_replica_route_hash` returns float32 and the check compares
`reduced_hash` against `local_hash * ep_size`. Values reach `modulus - 1 =
1,000,002`, so the product is exact in fp32 only while `ep_size <= 16`
(2^24 = 16,777,216). At larger EP the comparison can fail spuriously. That fails
loud rather than silent, so it is acceptable, but add an assert or a comment.

## 2.10 Two of the three compared cost policies are unreproducible — this is new

`_HBM_COST = 1280` / `_GRACE_COST = 3467` are **unchanged in both commits**:

```bash
git log --oneline -S "_GRACE_COST" -- vllm/model_executor/model_loader/tiered_moe_scheduler.py
# -> only 1a94ed458
```

There is no config field or env knob for the cost model. So the "interim M4/M16
scalar" (2.015/3.658, jobs `1128623`/`1128624`) and the "HBM-wave/Grace-chain"
policy (jobs `1130752`/`1130753`) were **uncommitted working-tree edits**. They
exist nowhere in the repository.

Three consequences:

1. The decision to keep the original scalar rests on a three-way comparison in
   which two arms cannot be inspected, re-run, or audited.
2. The README states the shape-dependent **replay** "keeps the original M=4
   scalar and uses the measured wave/chain costs at M=16". If the deployed chain
   runtime did the same, then **chain-c1 and original-c1 measured the same policy**
   — and their corrected-step results are -1.55% and -6.75%, a 5.2-point gap that
   would be pure between-job noise and would invalidate the ranking that rejected
   the chain model. Nothing in the repo can resolve this.
3. Anyone reproducing this experiment will silently run only the retained policy.

**Action, highest priority of this round:** make the cost model a
`TieredMoEConfig` field (or env) with the three variants as named options, commit
it, and re-state which variant each job used. Then either re-run the comparison
or mark its conclusion provisional.

## 2.11 The calibration is the best work in this round and its conclusion is premature

Jobs `1129855` and `1130891` produced a genuinely good physical model, replicated
across two nodes:

- HBM is **one ~167 us occupancy wave**, flat across 2-16 active experts
  (0.47% SD) — adding experts is free until the wave fills.
- Grace fits `max(167, 24 + 46 x active_experts)` us, reproducing within 0.4% on
  the second node (209/394/579/763 at 4/8/12/16) even though the HBM floor moved
  to 183-186 us.
- Varying routes-per-expert from 1 to 16 moved both tiers together, so token
  count is immaterial in this decode range — **this settles round 1's item 2.5
  sub-point about token-count sensitivity, negatively and with evidence.** Good.

The structural finding is a **breakpoint, not a ratio**: up to three Grace tasks
fit under one HBM wave; the fourth makes the Grace chain critical. A flat 2.709
scalar cannot express a breakpoint at all, so it cannot be the right model — it
can only be accidentally well-tuned for one operating point.

The conclusion "the isolated kernel physics does not justify a universal policy
replacement" is therefore stated too strongly, for two reasons: the end-to-end
comparison that rejected the chain model is unreproducible (2.10), and one of its
arms may have been identical to the baseline. Recommend restating as: *the chain
model is physically correct; the end-to-end comparison that rejected it is not
currently reproducible, and the deployed scalar is retained pending a
reproducible re-test.*

## Round 2 order of work

1. **Make the cost model configurable and commit the variants** (2.10). Without
   this, the central decision of this round is unauditable.
2. **Restate the status line on one metric** (2.6, 2.7) and add the
   acceptance-chaos explanation (2.8).
3. **Re-test the chain model** once (2.10, 2.11) with the variant selectable by
   config, three repeats, same node, corrected step as the metric. If chain and
   original really are identical at M=4, the c1 arms must agree — that is a free
   internal consistency check on the whole protocol.
4. Record the route-check limitations next to the feature (2.9).
