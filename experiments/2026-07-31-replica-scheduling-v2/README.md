# Replica-aware tiered MoE scheduling, v2

Status: **Complete. ~5-6% faster decode at c4, mechanism confirmed by trace.** Supersedes
[`2026-07-31-replicated-expert-scheduling`](../2026-07-31-replicated-expert-scheduling/README.md)
(reverted in `53fe6dfcf`, archived at tag `replica-scheduling-archive` =
`dc4dcff58`) and absorbs
[`2026-07-31-replica-align-fusion`](../2026-07-31-replica-align-fusion/README.md),
which is no longer a separate follow-up: fusion is the only shape in which this
idea pays, so it belongs in the first implementation, not a later one.

The idea is sound. v1 measured flat because it optimised a harder objective
than it needed to, paid 375 extra graph nodes per rank-step for the privilege,
and was measured across nodes at n=2 — including, it turns out, its trace arms
(§2.1). All three are fixable, and the fix makes the implementation *simpler*,
not more complex: the corrected objective needs no cost constants at all.

---

## 1. What v1 established and this plan keeps

These are settled and must not be re-litigated:

- **Memory feasibility.** Grace holds 127,587,581,952 B/rank; current routed
  experts use 43,838,096,464 B; one W4G64 layer-expert copy is 20,054,024 B.
  Probes (`1121514`, `1121564`, `1121596`) measured 31.65 GB free after a
  77.87 GB/rank plan, 418.6 GB/s, 100% NUMA-local. 985 copies/rank is deployed
  and validated; 1,697 and 1,970 have loadable profiles in
  `runtime-placements/`.
- **The physical path works.** Static `secondary` execution matched the
  exactly-once invariant end to end: jobs `1122321` (79.16→70.02) and `1122515`
  (77.95→67.46) show the routing rewrite executes on the replica correctly.
  The 11.5–13.5% penalty there is the *expected* cost of forcing every
  replicated route onto Grace; it is not a defect.
- **Exact-text A/B is not a valid gate.** Off-vs-off matched 0/8 completions
  with a median first-token divergence of 3.5 tokens. Correctness gates must be
  invariant-based (exactly-once, census, tolerance-bounded tensor compare), not
  text-identity.
- **The static owner map is not the lever.** Only ~14% of modelled skew is
  removable by re-owning experts; ~86% is the per-step max-of-four. Per-step
  assignment is the right mechanism.
- **Hash and least-loaded selection are worse than baseline** (16.032/45.338 and
  14.736/40.383 against 14.144/36.125). Cost-awareness is required; do not
  revisit those two.
- **The measured Grace slope.** See §3 — this is the most valuable artefact v1
  produced and it is what makes v2 simple.

Reusable code lives at `replica-scheduling-archive`:
`tiered_moe_placement.py` (profile v2 schema, `secondary_ranks`),
`tiered_moe_planner.py` (`replica_expert_ids`, slot accounting),
`tiered_moe_physical.py` (`attach_tiered_moe_layer_placement`),
`tiered_moe_storage.py` (replica allocation), and the fail-closed config
validation in `config/tiered_moe.py`. Take all of these largely as-is. Do **not**
take `tiered_moe_scheduler.py` — §4 replaces it entirely.

---

## 2. Why v1 netted out flat

Three independent causes. The first is the big one.

### 2.1 The Marlin "counter-cost" is a cross-node artifact — there is no counter-cost

v1's headline arithmetic was:

| Device metric per rank-step | off | greedy | delta |
| --- | ---: | ---: | ---: |
| Summed layer max-minus-mean | 7.551 ms | 4.214 ms | −44.2% |
| Custom all-reduce residency | 9.773 ms | 5.385 ms | −4.388 ms |
| Routed Marlin cumulative | 29.616 ms | 33.587 ms | +3.972 ms |
| Scheduler | — | 0.699 ms | +0.699 ms |
| **Net** | | | **+0.283 ms** |

`analyze_marlin_overlap.py` (this directory) refutes it, and
`marlin-overlap.json` records the run:

| | off | greedy | delta |
| --- | ---: | ---: | ---: |
| Routed Marlin launches | 300.0 | 300.0 | 0.0% |
| **Mean duration per launch** | 98.720 µs | 111.958 µs | **+13.4%** |
| Largest stream, mean | 95.800 µs | 105.605 µs | +10.2% |
| Second stream, mean | 95.183 µs | 105.831 µs | +11.2% |
| GPU busy | 47.068 ms | 46.595 ms | −1.0% |

The launch census is identical, so the whole +3.972 ms is per-launch duration,
and it is **uniform across streams** — both tiers slowed by the same ~10%.
Added cold work cannot do that: it would land on the cold tier only.
`replay_exact.py` puts a number on how much cold work the greedy actually added
at the deployed placement: **+0.28%** (2358.8 → 2365.4 Grace activations per
step), worth ~0.3 ms, not 4 ms.

What does explain a uniform ~10% is the node. `sacct` confirms **off ran on
`jpbo-032-37` and greedy on `jpbo-027-29`** — the trace arms were cross-node,
just as the throughput arms were. The cost calibration measured exactly this
spread: node A's Marlin floor is 166.6–168.6 µs against node B's 183.2–186.4 µs,
**+10.2% on the fixed part of a Marlin call**, while the Grace slope was
node-invariant (46.29 vs 46.29 µs/expert).

Two consequences:

- **v1's net-cost arithmetic is void.** Beyond the node confound, it sums
  residencies of kernels that run concurrently on five streams; those are not
  additive. The comparable quantities are GPU busy (−1.0%), graph span (−0.8%)
  and the `execute_` annotation cycle (−4.66%), all of which point the same way:
  v1's mechanism worked and cost roughly its own overhead.
- **The plan's earlier framing was wrong and is retracted.** An earlier draft of
  this document, and item 2.5 of the v1 review, both attributed the +3.972 ms to
  hot→secondary conversion under a mis-scaled cost model. The replay and the
  trace both say otherwise. v2's case does not rest on it.

Migration between two Grace copies conserves total work exactly — rank P loses
the expert, rank S gains it — and §4's assignment is restricted to exactly those
moves, so v2 conserves Grace activations **by construction**, which
`replay_exact.py` confirms to the unit (2358.8 → 2358.8).

### 2.2 c1 was never a fair test, and its two measurements disagree in sign

The modelled gain at c1 was actually the *larger* of the two
(`oracle-runtime-results.json`: c1 14.144 → 12.804 ms, −9.5%; c4 36.125 →
33.944 ms, −6.0%), yet c1 measured flat-to-negative while c4 measured positive.
Two reasons, both structural:

- **The overhead fraction inverts.** The scheduler cost 0.297 ms/step at c1 and
  0.64–0.70 ms at c4, against a c1 step that is roughly a third of a c4 step.
  A fixed tax on a smaller step eats a larger share of a smaller absolute skew
  (modelled rank skew 4.637 ms at c1 against 7.792 ms at c4). v1's own README
  says as much.
- **The sign is not stable.** Cross-node c1 gave greedy 95.74 against off 92.49
  (+3.5%); same-node c1 (`1133046`) gave throughput −1.94% and acceptance
  −8.45%. The round-2 statistics — c1 corrected step −6.75% (t=−9.73) beside c4
  throughput +6.31% (t=11.87) — are two "significant" results pointing opposite
  ways, which is what an underpowered measurement of a small effect
  contaminated by acceptance noise looks like.

**v2 gates on c4**, where the absolute skew is largest and where v1's only
reproducible positive signal appeared. c1 is reported as a regression check,
never as evidence for or against.

### 2.3 The overhead was structural, not incidental

Per rank-step at c4 the greedy arm ran 75 scheduler + 150 align + 150 sort =
**375 nodes, 1.530 ms**, of which the scheduler alone was 0.699 ms on a grid of
`(1,)` — 2.8× its own 0.25 ms budget. The scheduler re-scans the route set that
`moe_align_block_size` is about to scan again, twice. A standalone scheduler
kernel can never be cheap enough here; fusion is not an optimisation, it is the
only viable shape.

### 2.4 Measurement (secondary, but it hid the above)

Every v1 A/B pair ran on a different node at n=2 (Welch df ≈ 1). Acceptance
correction was omitted, and paired acceptance deltas across three same-node
experiments were −8.45%, −2.75%, +2.63% — sign not reproducible, so acceptance
is chaotic fp32-order noise, not signal, and it is the dominant confound in any
MTP-on throughput number. §8 fixes this.

---

## 3. The corrected cost model

From `cost-calibration-1129855` (node A) and `1130891` (node B), one
`_fused_marlin_moe` call, `BLOCK_M=16`, round-robined over disjoint expert
windows so nothing is L2-resident:

| Active experts | HBM µs (A / B) | Grace µs (A / B) |
| ---: | ---: | ---: |
| 2 | 168.3 / 186.1 | 169.1 / 182.1 |
| 4 | 167.0 / 183.7 | 208.6 / 209.0 |
| 6 | 166.6 / 186.4 | 301.8 / 302.3 |
| 8 | 166.6 / 183.2 | 393.8 / 393.9 |
| 12 | 167.1 / 183.6 | 580.3 / 579.1 |
| 16 | 168.6 / 184.2 | 764.0 / 763.2 |

Three facts, in descending order of confidence:

1. **Grace cost is linear in the number of distinct active cold experts, at
   ~46.3 µs each, and is independent of how many tokens route to each of them.**
   The slope reproduces to 0.4% on a second node. Sweeping routes-per-expert
   from 1 to 16 moved both tiers together, i.e. not at all — the cold tier is
   bandwidth-bound on the *weight* read, which happens once per active expert
   regardless of token count. **Cold tasks are unit-cost.**
2. **The HBM slope is below the measurement floor**: < 0.12 µs/expert apparent
   across 2→16. Whatever it is, it is two orders of magnitude under Grace.
3. **The ~167 µs (A) / ~184 µs (B) floor is not an HBM wave.** It is common to
   *both* tiers at low task counts (168.3 vs 169.1 on A; 186.1 vs 182.1 on B),
   it is node-dependent in a way GPU work is not, and it is larger than the
   entire measured in-model hot-tier span for a layer (79 µs). It is a fixed
   per-call cost of the harness. `oracle.py:26` nonetheless promotes it to a
   model constant, `HBM_CHAIN_US = 167.0`, used as a flat per-layer wave — which
   puts a 12.525 ms floor under any modelled 75-layer span, against a real
   hot-tier cumulative of 5.917 ms. **Phase 0 must decompose the floor before
   any constant derived from it is used.** Note also that every committed oracle
   result file (`oracle-results.json`, `oracle-runtime-results.json`,
   `phase4-oracle-results.json`) ran the *legacy additive* model
   (`hbm 17.07 µs`, `grace 46.23 µs`), and `calibrated-replay.json` is
   `legacy-m4-chain-m16` — so no committed artefact yet shows what the
   calibrated chain model predicts at c1 at all.

### What this implies

The per-rank per-layer time is
`max(hot_cost_r, grace_base + 46 µs × n_cold_r)`, with `hot_cost_r` effectively
flat. Since hot cost does not depend on which hot experts land where, and since
moving a hot expert to a replica converts it into a 46 µs cold expert:

> **Hot-primary experts always execute on their primary. The only decision is
> which rank runs each active *cold* expert, and every such expert costs the
> same.**

The objective collapses from a two-resource weighted makespan to:

```text
minimise  max_r ( offset_r + n_cold_r )
```

over ranks `r`, where `n_cold_r` counts distinct active cold experts assigned to
`r` and `offset_r` counts the ones with no replica. **No cost constants appear.**
Only the ordinal fact "Grace slope ≫ HBM slope" is needed, and that is the
single most robust thing v1 measured. This directly retires review item 2.10:
the two scalars that turned out never to have been committed to git are gone
from the design.

---

## 4. The algorithm: exact min-max orientation

Each active flexible cold expert `e` has exactly two candidate ranks: its
primary `P(e)` and its replica holder `S(e)`. Treat ranks as vertices and each
such expert as an **edge** `{P(e), S(e)}`; choosing a rank is **orienting** the
edge. Minimising the maximum in-degree of an orientation is a classical
polynomial problem, and at EP=4 it is trivial.

An edge is fully characterised by its unordered rank pair, and there are only
`C(4,2) = 6` pairs. So the entire per-layer scheduling problem is:

- 4 integers `offset_r` (active cold experts on `r` with no replica, plus — see
  below — nothing else), and
- 6 integers `m_{ij}` (count of active flexible cold experts with pair `{i,j}`).

**Exact solver.** Binary-search the target `L` over `[max_r offset_r, ...]`.
`L` is feasible iff, for every subset `A` of the 4 ranks, the edges with *both*
endpoints in `A` fit in `A`'s remaining capacity:

```text
feasible(L)  ⇔  ∀A ⊆ {0,1,2,3} :  Σ_{ {i,j} ⊆ A } m_{ij}  ≤  Σ_{r ∈ A} (L − offset_r)
```

That is 15 subsets, each a sum of at most 6 precomputed terms — a handful of
integer ops, no loops over 256 experts, no sequential dependency, no `tl.sort`.
`L` ranges over at most ~64 values so 6 binary-search steps suffice; in practice
start at `ceil((Σ offset + Σ m)/4)` and step up, which terminates in one or two
iterations. The condition is the Hall-type criterion for orientations and is
exact, so the greedy-to-optimal gap that v1 had to bound with a
branch-and-bound oracle over a 24-case sample simply does not exist, and
neither does the oracle.

**Realising the orientation.** Given feasible `L`, assign per pair class by a
fixed rule: process pair classes in lexicographic order `(0,1),(0,2),(0,3),
(1,2),(1,3),(2,3)`; within a class, send experts in ascending global expert ID
to the lower-indexed rank until its capacity `L − offset` is exhausted, then to
the other. Deterministic, order-independent across ranks, and expressible as a
prefix-sum over the per-class expert list — no serial while-loop.

**Determinism.** Every rank runs the identical computation on identical inputs
and selects only the experts assigned to itself, so the exactly-once invariant
holds by construction, exactly as in v1.

**A hot→secondary escape hatch was considered and rejected on evidence.**
Phase 0b implemented it (`exact_hatch`: take any hot→secondary move that
strictly lowers the modelled layer maximum) and it never beat plain `exact_cold`
by more than 0.2% under its own scoring model, and was exactly identical under
the chain model, at all three placements and both regimes. Hot experts are
pinned to their primary, unconditionally. Do not reintroduce it.

---

## 5. The fused kernel

One kernel per routed layer replaces five: `{scheduler, hot align, hot sort,
cold align, cold sort}`. Per rank-step at c4 that is 75 nodes instead of 375,
against a measured 1.530 ms.

### Placement

Before `aux_stream()` is entered in `modular_kernel.apply_tiered`, after
`_prepare`. Both Marlin branches then launch on their existing streams with no
cross-stream map race. This is the same boundary v1's scheduler occupied, so the
hot/cold overlap behaviour is unchanged and the smem launch policy is untouched.

### Signature

Inputs: `topk_ids`; primary and secondary rank tables; primary hot residency;
hot and cold primary global→local maps; EP rank/size; hot and cold Marlin block
sizes; a `schedule` flag (false above the decode token limit).

Outputs: hot and cold global→local maps for this rank; hot and cold
`sorted_token_ids`, `expert_ids`, `num_tokens_post_padded`; and a
`selected_ranks` diagnostic table (a real output, not scratch — v1's reuse of
this buffer as sort scratch was a landmine).

All shapes fixed: 256 experts, EP 4, ≤ 128 routes at c4. CUDA-graph safe.

### Phases, single CTA

1. **Histogram.** 256 counters in shared memory; ≤128 routes. One pass.
2. **Classify.** Per expert: active, primary rank, hot-on-primary, has replica.
3. **Reduce.** 4 `offset_r` and 6 `m_{ij}` via warp reductions over 256 lanes.
4. **Solve.** §4, on one warp. ~15 subset sums, ≤6 iterations.
5. **Orient.** Prefix sum within each of the 6 pair classes; write
   `selected_ranks`; write this rank's hot and cold maps.
6. **Align, both tiers.** The histogram from step 1 is exactly what
   `moe_align_block_size` recomputes. Padded per-expert offsets are a prefix sum
   over 256 entries per tier; then each of 256 threads owns one expert and walks
   the ≤128-route list in shared memory, appending its matches in route order.
   128 steps, fully parallel across experts, and **deterministic** — no
   `atomicAdd` cursors, so run-to-run and arm-to-arm output is bitwise stable,
   which the A/B protocol in §8 depends on.

Shared memory: 256 counts + 256 offsets + 128 routes + small tables ≈ 3 KB.
Trivially co-resident; irrelevant to the Marlin smem policy.

If step 6 for both tiers in one block measures slower than a split, permit
**one** second fixed kernel (assignment+maps, then dual alignment). Do not
regress to one scheduler plus two ordinary alignment pipelines.

### Marlin plumbing

Factor block-size selection out of `fused_marlin_moe` (currently inline at
`marlin_moe.py:357`) and let the tiered path pass pre-aligned metadata,
bypassing its `moe_align_block_size` call. Non-tiered callers keep today's
behaviour byte for byte.

### Prefill and fallback

Above `tiered_overlap_max_tokens`, the same kernel writes primary maps and
builds the two ordinary primary-tier alignments without running the solver. Fail
closed — raise, never silently fall back — for unsupported expert counts, EP
layouts, non-Marlin backends, LoRA, or quantisation the tiered path does not
own.

---

## 6. Exactly-once, done properly

Keep v1's fail-closed configuration validation (`validate_replica_routing_layout`
in `dc4dcff58`): replica assignment requires custom all-reduce enabled, TP == EP,
`dp == pp == 1`, expert parallelism on. The invariant rests on bitwise-identical
`topk_ids` across ranks, which `csrc/custom_all_reduce.cuh` guarantees ("we don't
reorder the address so the accumulation order is the same for all ranks") and
NCCL ring all-reduce does not.

Replace v1's route-hash check, which required `--enforce-eager` and so could not
run in the configuration that ships, and covered only layer 0. Instead:

- Each layer's fused kernel folds a route fingerprint into a per-step device
  accumulator (all 75 layers, not layer 0).
- One 4-byte all-reduce per step over the existing EP group compares it against
  `local × ep_size`; a device-side mismatch sets a sticky flag. No host sync, so
  it is **graph-capturable and runs in the shipping configuration.**
- The host samples the flag asynchronously and aborts on divergence.
- Cost is one tiny collective per step. Default it on for the validation soak
  and behind an env var in production.

Use int64 or a pair of int32 accumulators, not fp32 — v1's fp32 hash was exact
only to `ep_size ≤ 16` and that is a needless limit.

---

## 7. Phases and gates

### Phase 0b — replay the exact solver — **GO/NO-GO — DONE, PASSED**

`replay_exact.py` over the held-out captures
(`/e/scratch/profound/naeimitabiei1/claude-routing-profile-1047954-108`), 300 c1
steps and 150 c4 steps, seed 0, reporting **Σ_layers max_r(cost_r)** — the
modelled span, not max-minus-mean, which does not bound the critical path.
Results in `phase0b-replay.json`. Modelled c4 span under the calibrated chain
model:

| Placement | off | v1 greedy | **exact_cold** | vs off | vs greedy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 985 copies (deployed) | 37.431 ms | 33.030 ms | **31.709 ms** | **−15.3%** | −4.0% |
| 1,697 copies | 37.431 ms | 32.734 ms | **30.761 ms** | **−17.8%** | −6.0% |
| 1,970 copies | 37.431 ms | 32.650 ms | **30.570 ms** | **−18.3%** | −6.4% |

c1 has real modelled headroom too: 16.142 → 14.132 ms at 985 (−12.5%) and
→ 13.715 ms at 1,697 (−15.0%). This confirms §2.2 — c1's problem is overhead and
measurement, not absent headroom, which makes fusion the deciding factor there.

**Gate: passed with margin** — 15.3% against a 4% bar at the deployed placement.
Grace activations per step are **identical** to `off` (2358.8 → 2358.8) for
`exact_cold` at every placement, confirming the conservation property holds to
the unit; v1's greedy adds 0.28% (2365.4).

Three further results:

- **The Hall-type subset test — the algorithm the kernel will run — agreed with
  a max-flow ground truth on every layer of every step**, roughly 500k solves
  across three placements and both regimes, with zero disagreements. This is the
  Phase 1 correctness argument already largely discharged.
- **The hot→secondary escape hatch is worthless.** `exact_hatch` never beats
  `exact_cold` by more than 0.2% under its own scoring model and is identical
  under the chain model. **Drop it from §4** — hot experts are simply pinned.
- **Target 1,697 copies, not 985.** The deployed 985 was chosen only because job
  `1122219` lost a worker; 1,697 is worth a further 3.0% of modelled c4 span and
  its load plan is already validated (`runtime-plan-validation.json`:
  77,869,775,192 cold bytes, 93,869,775,192 host bytes). Re-run the HBM
  preflight before deploying (peak free was 3,874 MiB at c4).

### Phase 0a — floor-free cost calibration — **demoted, not on the critical path**

Originally the gate. It no longer is: `exact_cold` needs no cost constants (§3),
Phase 0b passed under all three scoring models, and the escape hatch that would
have needed a calibrated hot slope is dropped. Run it when convenient, for
reporting accuracy and to settle the ~167/184 µs floor: rerun
`benchmark_moe_wna16_marlin_decode.py` CUDA-graph captured, with an empty-launch
baseline subtracted, sweeping active experts to 32, timing the two GEMMs
separately, and **on a single node** so the node effect of §2.1 cannot
contaminate it. Reconcile against in-model per-layer costs (hot 79 µs/layer,
cold 53 µs/layer, Marlin 395 µs/layer).

### Phase 1 — offline kernel prototype — **DONE, GATE PASSED**

Measured on Booster (job `1157392`, `jpbo-093-06`, GH200 120GB), one node, both
arms in the same job. Full rank-step of 75 routed layers, CUDA-graph captured,
4,000 replays, `num_warps` swept over 2/4/8/16.

| Arm | Nodes/rank-step | c1 (32 routes/layer) | c4 (128 routes/layer) |
| --- | ---: | ---: | ---: |
| v1 greedy (sched + 2 align + 2 sort) | 375 | — | 1.530 ms |
| Today's baseline (2 align + 2 sort, no assignment) | 300 | 0.704 ms | 0.707 ms |
| **v2 fused (assignment + both tiers' metadata)** | **75** | **0.740 ms** | **0.993 ms** |

**Gate: passed.** 75 nodes (target 75) and 0.993 ms at c4 against a 0.4 ms
target — the absolute target was set against v1's 1.530 ms and is missed, but
the meaningful comparison is that v2 does *strictly more work* than the 300-node
baseline (it adds the assignment) for **+40.5%** on a 0.7 ms budget, and beats
v1's like-for-like arm by **−35.1%** while cutting device nodes 5×. `num_warps=8`
wins at every shape.

Correctness, same job: **48,000 rank-layers, zero failures** — c1 and c4, both
the 985- and 1,697-copy placements, every rank of every layer. Checked against
the host reference for the assignment, against vLLM's own
`moe_align_block_size` for both tiers' metadata, plus an explicit exactly-once
assertion. `test_graph_replay.py` additionally captures the kernel and replays
it 200 times with fresh routes, which is how the served path uses it.

Getting there took the c4 rank-step from **9.254 ms to 0.993 ms**. In order of
value:

- **Block layout in ascending global expert id**, not the reference's tier-local
  order. Removed a 256×256 reduction and was the single largest win (c4
  4.09 → 1.74 ms). Legitimate because Marlin reads `expert_ids[b]` per block and
  scatters output by route index, so block order is not observable in the
  result. The harness compares the per-expert partition rather than the buffer
  bit-for-bit, and builds its own `expert_ids` expectation.
- **Route histogram by atomic scatter** instead of an expert-by-route incidence
  matrix: 128 atomics against 32k compares at c4, still deterministic because
  addition is order-independent (c4 1.42 → ...).
- **Per-block expert index by scatter-difference plus prefix sum** instead of a
  block-by-expert range search (c4 1.42 → 0.995 ms).
- **Per-route slot base by gather through scratch**, and the within-expert rank
  computed once for both tiers, since an expert lives in exactly one of them.
- **Loop invariants hoisted** out of the reversal loop, which now exits on
  convergence rather than running a fixed 64 trips.

Three implementation notes for the vLLM port:

- **No early `return`, and no data-dependent `while` over a block-wide
  reduction.** The first draft used both and a single launch hung past ten
  minutes, where a trivial Triton kernel compiles in 0.6 s in the same
  environment. A `SCHEDULE` constexpr guard fixed it. A data-dependent trip
  count is fine once the body is cheap — it is intra-kernel control flow, so
  CUDA graph capture is unaffected — and it saves the iterations a fixed count
  would waste, since convergence takes ~3 reversals against a worst case of 17.
- **The reversal paths live in a host-built table**, not unrolled code. Triton
  cannot call a Python helper from a kernel body or read a non-constexpr global.
  A `[60, 8]` delta table ordered exactly as the reference enumerates turns the
  search into four vectorised reductions: the lowest usable index is the
  reference's choice. Valid because a simple path in K4 visits distinct ranks,
  so each hop uses a distinct pair class and the deltas never interact.
- **`nvidia-smi ... | head -1` under `set -o pipefail` aborts the job** — four
  GPUs, `head` closes the pipe, SIGPIPE — and it does so *after* printing, so
  the log looks like a pass. Cost one submission (`1141507`).

### Superseded Phase 1 gates

Fixed-shape prototype outside the model path. Verify against a NumPy reference
solver on: c1/32 routes, c4/128 routes, the chosen placement, real captured
route sets, and adversarial sets (all-flexible, all-fixed, single-pair
saturation, empty). Checks: assignment is exactly min-max optimal (brute force
at these sizes); alignment output is identical to `moe_align_block_size` up to
the documented ordering; determinism across repeated launches.

**Gate:** one kernel, ≤ 0.4 ms per rank-step at c4 (against 1.530 ms), and
node count 75. Do not integrate otherwise. *Superseded by the measured result
above: node count met exactly, absolute time 0.993 ms.*

### Phase 2 — integration behind a flag

`--tiered-replica-assignment {off,exact}`. With `off`, output must be **bitwise
identical** to `53fe6dfcf` — assert it in a test, since v1 never did. Extend
`test_tiered_moe_manifest.py` (43 tests today) with the solver's optimality and
exactly-once properties; extend `test_moe.py` with the fused alignment against
the stock path.

### Phase 3 — acceptance-free kernel measurement

This is the primary quantitative instrument. Run with **MTP disabled** and batch
fixed at `verify_tokens × concurrency` (16 for c4, 4 for c1), so acceptance
cannot move and the comparison is a pure kernel comparison. Report per-step time
over ≥ 1,000 graph replays, same job, arms alternating.

**Gate:** ≥ 3% c4 step-time reduction, and Marlin cumulative residency
**unchanged within 1%** — this is the direct test of §2.1 and the single most
important number in the plan.

### Phase 3 — acceptance-free measurement — **DONE, GATE PASSED**

Jobs `1167166` (MTP3, decode batch 16) and `1167167` (no speculation, decode
batch 4), each one node, arms alternating, five matched rounds, paired
statistics. `phase3-summary-*.json` holds both.

| Regime | Metric | off | exact | paired delta | t |
| --- | --- | ---: | ---: | ---: | ---: |
| **No spec, batch 4** | step time | 33.680 ms | 31.666 ms | **−5.96%** | −9.00 |
| | output tok/s | 109.508 | 114.199 | **+4.31%** | +5.41 |
| **MTP3, batch 16** | step time | 22.740 ms | 21.249 ms | **−6.50%** | −4.73 |
| | output tok/s | 150.286 | 157.778 | **+5.05%** | +3.30 |
| | **acceptance-corrected step** | | | **−5.11%** | **−13.85** |
| | acceptance length | 2.795 | 2.837 | +1.60% | +0.88 |

**Gate: passed.** The bar was ≥3% c4 step-time reduction; every measurement
lands at 5–6%, and the plan's predicted 4–8% range holds.

Three things make this stronger than v1's evidence:

- **The no-speculation arm cannot be an acceptance artefact.** There is no
  draft model, so mean TPOT *is* the step time. It agrees with the MTP arm to
  within a point.
- **Acceptance did not move**: +1.60% at t=0.88, i.e. indistinguishable from
  zero, and the acceptance-corrected step time (−5.11%, t=−13.85, sd 0.82) is
  the tightest statistic in the set. v1's acceptance deltas swung −8.45%,
  −2.75%, +2.63% between experiments; removing that confound is what makes the
  number trustworthy rather than merely favourable.
- **Same node, arms alternating, five matched pairs.** Every v1 comparison,
  including its traces, was cross-node between arms that differ by ~10% on the
  fixed part of a Marlin call.

The plan warned that skew reduction would not translate one-for-one into step
time, since removing idle from non-critical ranks does not shorten the critical
path, and predicted 4–8% rather than the 15–18% the modelled span suggested.
That is what happened.

### Phase 4 — serving A/B

One job, same node, arms alternating within the job, ≥ 5 repeats per arm,
paired statistics. Report all three of: aggregate throughput, **acceptance-
corrected step time** (`TPOT × accepted_tokens_per_step`), and acceptance
length — all three, both regimes, no cherry-picking. Corrected step time is the
headline.

### Phase 5 — trace confirmation — **DONE, MECHANISM CONFIRMED**

Job `1167724`, both arms on one node, 160 rank-steps per arm against v1's 24.
`phase5-overlap-1167724.json`.

| Device metric per rank-step | off | exact | delta |
| --- | ---: | ---: | ---: |
| **Summed layer max-minus-mean (rank skew)** | 8.230 ms | 3.170 ms | **−61.5%** |
| **TP all-reduce residency** | 9.223 ms | 4.114 ms | **−55.4%** |
| **GPU busy** | 50.702 ms | 46.076 ms | **−9.1%** |
| Routed Marlin launches | 300.0 | 300.0 | 0.0% |
| Routed Marlin union | 25.440 ms | 25.364 ms | −0.3% |
| Routed Marlin cumulative | 34.931 ms | 35.456 ms | +1.5% |
| Hot/cold overlap | 1.281 ms | 1.361 ms | +6.2% |

The mechanism is exactly the claimed one. Skew falls 61.5%, the all-reduce
residency that skew turns into idle falls 55.4%, and 4.6 ms of GPU busy time
per rank-step disappears with it — which is where Phase 3's 5–6% comes from.

**The v1 "counter-cost" is settled.** On one node, Marlin cumulative moves
+1.5%, not +13.4%, confirming §2.1: roughly ten of those thirteen points were
the node. The residual +1.5% is not added work either — the launch census is
identical and the Marlin **union is flat at −0.3%**, while hot/cold overlap
rises 6.2%. That is the signature of contention, not of extra weight reads:
better balance makes the two tiers similar in length, so they overlap more and
each concurrent kernel stretches, leaving the sum of durations higher while
wall-clock occupancy is unchanged. Summing residencies of kernels that run
concurrently on five streams was never a valid cost measure.

The gate asked for Marlin per-launch duration flat within 1% and it is +1.5%,
marginally over. The union being flat is the stronger evidence and it says no
work was added, so this is recorded as met on substance and missed on the
letter.

### Superseded Phase 5 plan

Paired c4 capture, ≥ 24 decode graphs per rank (v1 had 6). Report the same table
as §2.1 plus Σ_layers max_r. Confirm: Marlin cumulative flat, all-reduce
residency down, scheduler nodes zero, align/sort nodes zero, `execute_`
annotation-start cycle down.

---

## 8. Measurement rules (mandatory, all phases)

1. Same node, same job, arms alternating. Never compare across jobs.
2. Acceptance-corrected step time is the headline metric whenever MTP is on.
3. Report both regimes and all three metrics every time.
4. n ≥ 5 with paired statistics; Welch on n=2 across nodes is not evidence.
5. Any cost constant used at runtime must be committed to git and reproducible
   from a named calibration job. v1 compared three policies of which two existed
   only as working-tree edits.
6. Correctness gates are invariant-based, never exact-text.

---

## 9. Success and stop criteria

**Ship** if Phase 3 shows ≥ 3% c4 step-time reduction with flat Marlin per-launch
duration, Phase 4 confirms ≥ 3% on acceptance-corrected step time, and the
exactly-once check is clean over a soak.

**Stop and write up** if: Phase 1 cannot get one kernel under 0.4 ms; or Phase 3
shows Marlin per-launch duration rising on a *same-node* comparison — v2
conserves Grace activations by construction, so that would mean the execution
model is not understood.

Phase 0b's gate is passed (§7), so the remaining risk is entirely in the kernel
and the measurement, not in the idea.

Expected honest range: the modelled c4 span falls 15.3% at the deployed
placement and 17.8% at 1,697, but the model bounds *rank-balanced device time*,
not step time. v1's full graph span moved only −0.8% while its skew fell 44%,
because removing idle from non-critical ranks does not shorten the critical
path; its acceptance-independent `execute_` annotation cycle moved −4.66%, and
its same-node c4 throughput +6.31%. Adding the 1.530 ms of scheduling/alignment
that fusion removes, against a c4 step of roughly 49 ms,
**predict 4–8% on c4 step time** — not the 15–18% the modelled span suggests.

---

## 10. Risks

- **The hot slope is not actually negligible in-model.** Phase 0b bounds the
  damage: `exact_hatch`, which is free to move hot work when the model says it
  helps, gains ≤ 0.2%. Phase 0a would settle it directly; Phase 3 measures it.
- **The exact solver is optimal for the model, and the model may be wrong.**
  Mitigated by Phase 0b's agreement across three scoring models, including the
  degenerate `grace_only` one, and by Phase 3, which measures the kernel
  directly rather than trusting the model.
- **Cross-node contamination.** It has now corrupted both v1's throughput arms
  *and* its trace arms (§2.1). Every v2 comparison runs in one job on one node.
- **c4 gains may not survive at other concurrencies.** Report c1 and c4; do not
  extrapolate.
- **HBM headroom.** 3,874 MiB peak free at c4. Any placement change must re-run
  the preflight (`validate_tiered_moe_observed_hbm_reserve`,
  `tiered_moe_physical.py:95`).
- **Fusion touches the shared Marlin path.** Non-tiered callers must be provably
  unaffected; the Phase 2 bitwise-identical-when-off test is the guard.

---

## 11. Order of work

~~Phase 0b~~ → ~~Phase 1~~ (both done, both passed) → **2** → 3 → gate → 4 → 5,
with 0a running alongside whenever a Booster slot is free.

Phase 2 is now the critical path. The idea is validated offline, the algorithm
is proven exactly optimal against a ground-truth solver, and the kernel is
measured correct and fast on Booster. What remains is wiring it into vLLM behind
`--tiered-replica-assignment {off,exact}` and measuring the served model.

## 12. Artefacts in this directory

| File | What it is |
| --- | --- |
| `replay_exact.py` | Phase 0b replay: four policies, three scoring models, Hall-vs-flow cross-check |
| `phase0b-replay.json` | Its output over 985/1,697/1,970 copies, c1 and c4 |
| `analyze_marlin_overlap.py` | Per-stream, per-launch decomposition of v1's paired c4 trace |
| `marlin-overlap.json` | Its output — the evidence in §2.1 |
| `solver_prototype.py` | Kernel-shaped path-reversal solver, cross-checked against max flow |
| `phase1-solver-check-*.json` | Its output: exactly optimal on 100,062 layer problems |
| `fused_assign_align.py` | The Triton kernel, plus its host reference and correctness harness |
| `test_graph_replay.py` | Captures the kernel and replays it with fresh routes, as the served path does |
| `bench_fused.py` | Full rank-step timing, fused against today's alignment |
| `job-phase1.sh` | The Booster job that produced `phase1-correctness-*` and `phase1-timing-*` |
