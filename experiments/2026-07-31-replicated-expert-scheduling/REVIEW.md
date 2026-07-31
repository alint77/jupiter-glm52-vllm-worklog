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
