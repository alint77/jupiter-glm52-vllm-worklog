# Replicated Grace experts with per-step makespan scheduling

Status: **implementation plan; no runtime changes or performance claims yet**.

## Goal

Use otherwise free paired-Grace memory to keep selected routed-expert weights on
more than one EP rank. After routing, assign each active logical expert to
exactly one available physical copy so that the predicted slowest-rank MoE time
is minimized.

The target is the synchronization time caused by per-step expert variance, not
the communication payload itself. The existing TP all-reduce remains the
correct final reduction and should keep the same count and payload.

## Why this is the right problem

Two independent trace analyses point to the same opportunity:

- The short-context c1/MTP3 post-fix trace has **4.128 ms/step** of
  communication-kernel residency above the fast-rank payload/protocol proxy.
  The target model's 157 custom all-reduces contain nearly all of it.
- The c4/400K/MTP3 critical-path trace attributes **7.82 ms/step** of
  post-MoE all-reduce wait to expert skew. Replaying captured routes showed
  that only about 14% of its modelled skew is removable with a static owner
  map; about 86% is the per-step max-of-four-ranks effect.

A better static owner map cannot remove variance in which experts co-activate
in one step. Replicas create a choice after the route set is known.

## Memory feasibility

The current physical plan reports:

| Quantity | Per rank |
| --- | ---: |
| Grace capacity | 127,587,581,952 B |
| Current Grace expert allocation | 31,609,860,800 B |
| Current configured host reserve | 8,000,000,000 B |
| One runtime layer-expert copy | 19,464,200 B |
| Existing ownership | 64 experts/layer × 75 layers |

With only the current 8 GB reserve, the arithmetic permits 4,519 extra
layer-expert copies per rank: 60.25 per layer. A rank would hold about 124 of
256 experts per layer on average, and the system would have 1.94 physical
copies per logical layer-expert. If no expert receives more than one secondary
copy, approximately 94% could have a second home.

That is a capacity ceiling, not the initial operating point. The experiment
will default to a **16 GB hard Grace reserve**, derived from the complete
physical plan rather than hard-coded. With the current numbers this permits:

| Grace reserve | Extra copies/rank | Extra/layer/rank | System copy factor |
| ---: | ---: | ---: | ---: |
| 16 GB | 4,108 | 54.77 | 1.856× |
| 24 GB | 3,697 | 49.29 | 1.770× |

The planner must include any host KV cache and other fixed allocations before
computing this budget. Large pinned-UVA allocations can fail below the simple
capacity limit, so an allocation-and-touch probe is a gate before model work.

## Terminology and invariant

- A **logical expert** is one of the model's 256 experts in a layer.
- A **primary copy** is the rank that owns the expert today.
- A **secondary copy** is an additional Grace-resident copy on another rank.
- An **expert task** is all routes selecting one logical expert in one layer
  and engine step.

The non-negotiable correctness invariant is:

> Every valid token/expert route is executed by exactly one rank that stores
> the expert, with its original router weight.

If two replicas execute the same route, the current all-reduce double-counts
the output. If no replica executes it, output is lost.

## Chosen architecture

### Static replica placement

Keep the current primary ownership and hot/cold residency unchanged. Add only
Grace-resident secondary copies. This gives an exact off control: when replica
scheduling is disabled, loading replicas cannot change execution.

For each layer, every rank receives the same small immutable tables:

- `rank_mask[logical_expert]`: ranks holding a copy;
- `tier[logical_expert, rank]`: absent, HBM, or Grace;
- the calibrated incremental cost of using each copy.

Secondary placement must:

1. respect the exact per-rank Grace budget and reserve;
2. never place the same logical expert twice on one rank;
3. preserve at least one copy of every logical expert;
4. balance secondary slots and primary-secondary rank pairs;
5. prioritize experts with high co-activation and straggler value, not merely
   global frequency.

At the safe budget, most useful experts can receive one secondary copy. More
than two total copies should be allowed by the format but is not part of the
first implementation.

### Dynamic assignment

All ranks in the current TP4/EP4 path receive the same logical routing result.
A deterministic GPU operation can therefore produce the same assignment on
every rank without a new collective.

The scheduler runs after logical top-k selection and before the existing local
expert map. It returns the selected rank for each expert task. Each rank masks
routes assigned elsewhere, then the existing tiered prepare, Marlin execution,
finalize, and TP all-reduce proceed normally.

The first implementation will assign an entire logical expert task to one
rank. It will not split different tokens for the same expert between replicas.
Decode Marlin is dominated by reading expert weights, so splitting usually
duplicates the most expensive work and harms batching.

The assignment must be:

- GPU-side and CUDA-graph safe;
- allocation-free with fixed-shape buffers;
- deterministic, including ties;
- independent of unsynchronized wall-clock measurements;
- enabled only for an execution layout whose rank route IDs have been
  validated as identical.

### Why not directly enable existing EPLB

vLLM EPLB already models logical experts, redundant physical experts, and
logical-to-physical maps. It is useful reference code, but its current router
selects a replica using a token hash. It does not minimize the current
four-rank makespan.

The tiered path also currently assumes exactly 64 uniquely owned experts per
rank and divides them between HBM and Grace. Adopting all EPLB state,
rearrangement, and equal-physical-slot machinery would expand the first
experiment substantially.

The minimal first path is therefore a tiered-specific logical-expert assignment
mask. EPLB mapping and tests should be reused where they reduce code, but
dynamic expert rearrangement and weight migration remain out of scope.
Tiered replica scheduling and generic EPLB should fail closed if configured
together during this experiment.

## Scheduling objective

For rank \(r\), maintain predicted HBM and Grace chain costs after assigning
some active expert tasks:

```text
rank_time[r] = fixed_rank_cost[r] + overlap(hbm_cost[r], grace_cost[r])
objective    = minimize max(rank_time[0:4])
```

The initial overlap model is:

```text
overlap(hbm, grace) = max(hbm, grace)
```

It will be replaced by a measured two-dimensional lookup only if that simple
model fails validation. The tight-shared-memory work allows overlap, but actual
contention means neither a pure sum nor a pure maximum should be assumed
without checking.

The incremental expert cost must depend on:

- HBM versus Grace residence;
- target batch shape, initially M=4 and M=16;
- distinct active expert count;
- token count for that expert where it materially changes cost.

### Online algorithm

An exact combinatorial search per layer would cost more than the work it saves.
The online scheduler will use deterministic longest-processing-time greedy:

1. count routes and form the active unique expert tasks;
2. order tasks by descending worst candidate cost, then logical expert ID;
3. for each task, choose the available rank giving the smallest resulting
   global maximum;
4. break equal choices by rank ID.

A single bounded improvement pass may be added only if offline results show a
meaningful gap. The final implementation should fuse counting and assignment
into existing routing work where practical; a separate kernel is acceptable
for the proof of concept.

### Offline oracle

Before runtime changes, solve captured steps offline using:

1. current unique ownership;
2. replicated placement with hash selection;
3. replicated placement with token-count least-loaded selection;
4. cost-aware greedy makespan selection;
5. a branch-and-bound oracle on small steps.

The oracle establishes the available gain and the greedy-to-optimal gap. If it
predicts little improvement, the runtime implementation stops here.

## Experimental phases

### Phase 0 — capture and calibrate

The existing hotness grid is aggregated and loses which experts co-activated
in a particular layer and step. Extend the opt-in capture path to store compact
per-step route sets:

```text
(request/step ID, layer ID, target or draft, M, logical top-k IDs)
```

Collect realistic Python, PyTorch, CUDA, machine-learning, math, and email
prompts at c1 and c4. Include all target verification positions used by MTP3.
Do not use a single 256-token synthetic completion as the placement dataset.

Calibrate HBM and Grace expert-task costs from existing kernel benchmarks and
matched traces. Validate predicted per-rank routed-layer time against a control
trace before using the model for scheduling.

Gate:

- predicted per-layer rank time error no worse than 15% in aggregate;
- captured rank route tables must match exactly in the current execution path.

### Phase 1 — offline placement and assignment oracle

Evaluate replica budgets corresponding to roughly 25%, 50%, 75%, and 90% of
currently free Grace capacity. The current-plan examples are:

| Fraction of theoretical extra space | Extra copies/rank | Extra GB/rank | Copy factor |
| ---: | ---: | ---: | ---: |
| 25% | 1,129 | 21.98 | 1.235× |
| 50% | 2,259 | 43.97 | 1.471× |
| 75% | 3,389 | 65.96 | 1.706× |
| 90% | 4,067 | 79.16 | 1.847× |

For every budget, jointly choose secondary placement and replay assignments.
Report:

- predicted `sum_layer max_rank`;
- predicted all-reduce rank-wait reduction;
- HBM-to-Grace task migrations;
- selected-copy residence distribution;
- greedy versus oracle gap;
- sensitivity across prompt domains.

Go gate: predict at least **0.75 ms/step** reduction in the realistic c1
regime or **2.0 ms/step** in the c4 stress regime after charging estimated
scheduler overhead.

### Phase 2 — physical allocation and loader

First run a four-rank Grace allocation, registration, first-touch, and C2C-read
probe at the candidate budgets. Preserve NUMA-local allocation and measure
available Grace memory after allocation.

Then extend the tiered physical plan and streaming loader:

- permit duplicated ownership across ranks and variable local counts;
- retain uniqueness within one rank;
- budget replicas as cold expert bytes;
- load and convert secondary experts directly into final Grace storage;
- keep conversion scratch bounded to one expert;
- emit a cross-rank coverage and capacity summary;
- fail closed if any logical expert lacks a primary or any rank exceeds plan.

Loading may read almost twice as many expert weights from the checkpoint.
Measure startup time and shared-filesystem bandwidth, but treat it as a
secondary serving concern.

### Phase 3 — correctness-first routing

Implement a minimal deterministic replica selector with the cost-aware
scheduler disabled. Use a static secondary choice or existing hash idea to
exercise masking and reduction semantics.

Required checks:

- each valid route is assigned exactly once;
- assignments only target available copies;
- all ranks independently produce identical assignments;
- summed replica output matches unique ownership;
- disabling scheduling is bitwise equal to the existing path;
- realistic greedy decoding produces identical tokens and acceptance
  decisions.

No performance conclusion is allowed from this phase.

### Phase 4 — GPU makespan scheduler

Add the minimal graph-captured greedy scheduler and its static cost tables.
Measure it in isolation and in the whole model. Avoid CPU synchronization,
per-layer allocations, dynamic Python control flow, or a general scheduling
framework.

Overhead gate:

- no new CPU round trip;
- at most 0.25 ms total scheduler cost over all 75 target layers;
- no change in TP all-reduce count or payload;
- no regression when no selected expert has a useful replica.

If a separate kernel consumes too much, fuse selection into the existing
route-map/count stage before considering a more sophisticated algorithm.

### Phase 5 — performance and trace validation

Run matched controls and candidates in parallel, using separate Booster nodes:

1. current unique ownership;
2. replicas loaded but scheduling disabled;
3. hash replica selection;
4. cost-aware greedy selection;
5. best lower-memory replication budget if different.

Primary workload:

- the established realistic mixed prompt suite;
- c1 and c4;
- AutoRound W4G64;
- MTP3;
- identical seeds, request ordering, context, and output limits.

Use the c4/400K case only as a stress/profile diagnostic, not as the sole
throughput result.

Capture a four-rank trace for control and the best candidate. Compare:

- mean and confidence interval of output tokens/s;
- aggregate c4 tokens/s and per-request decode latency;
- MTP acceptance;
- Grace and HBM headroom per rank;
- custom all-reduce observed residency and fast-rank floor;
- per-layer rank arrival spread;
- routed hot/cold spans per rank;
- scheduler kernels and cost;
- selected primary/secondary and HBM/Grace rates.

The desired signature is lower custom-all-reduce residency with an unchanged
payload floor. A lower reported barrier duration without lower step wall is not
a success.

## Test design

The test module covers replicated placement and assignment. Its contract is:

- **Module purpose:** choose and execute one valid physical copy per logical
  route under fixed memory and availability constraints.
- **Inputs:** logical top-k IDs, replica/tier maps, deterministic cost tables,
  rank, and fixed-size work buffers.
- **Outputs:** selected rank or locally masked top-k IDs, plus optional
  assignment counters.
- **Primary failure guarded against:** duplicate or missing expert output.
- **Cheapest useful levels:** pure planner/scheduler unit tests first, a small
  four-rank reduction test second, then full-model evaluation.

Extend
`tests/model_executor/model_loader/test_tiered_moe_manifest.py` for planning,
loading, and tiered execution behavior. Reuse EPLB fixtures and mapping tests
where they fit. Add a new focused scheduler test file only if the GPU scheduler
has no natural nearby suite.

Minimum behavioral cases:

- exact Grace capacity boundary and reserve;
- duplicated cross-rank ownership accepted, same-rank duplicate rejected;
- every logical expert covered at least once;
- deterministic ties;
- adversarial route set where greedy moves a task from the overloaded primary;
- hot primary preferred until its predicted makespan exceeds a cold secondary;
- repeated routes for one expert remain one task;
- invalid, padded, and empty routes;
- CUDA graph capture and replay stability;
- BF16 output equality against unique ownership.

Because this changes model execution, run the established semantic prompt
suite and record evaluation results, not only unit tests.

## Likely implementation surface

Keep the first patch local to the tiered path:

- `tiered_moe_placement.py`: replica profile schema and cross-rank validation;
- `tiered_moe_planner.py`: exact duplicate ownership and Grace budgeting;
- `tiered_moe_runtime.py`: variable local-count map sizing;
- `tiered_moe_storage.py` and streaming loader: secondary final storage;
- `tiered_moe_execution.py`: assignment before prepare;
- one small GPU assignment implementation near tiered execution or routing;
- existing tiered manifest/planner tests and experiment benchmarks.

Do not alter generic EP/EPLB behavior until the experiment demonstrates a
benefit and the human submitter chooses an upstreamable abstraction.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Duplicate output | Exactly-one assignment assertions and summed-output tests |
| Scheduler costs more than skew saved | Offline gate, simple greedy, 0.25 ms budget |
| Cold replica is slower than overloaded hot primary | Residence-aware incremental costs |
| Cost model misses tier contention | Validate against traces; introduce a small lookup only if needed |
| Pinned Grace allocation fails | Allocation/touch probe and 16–24 GB reserve |
| Domain-specific placement | Train on mixed actual usage; report domain slices |
| CUDA graph instability | Fixed maps/buffers and capture/replay tests |
| Checkpoint startup becomes excessive | Measure separately; keep serving TPS as primary |
| Future sharded route tables differ across ranks | Restrict the feature to validated layouts; fail closed rather than add a dispatch collective |

## Success and stop criteria

Proceed beyond the proof of concept only if:

- output correctness and MTP behavior are unchanged;
- the scheduler stays within its 0.25 ms/step budget;
- physical Grace headroom remains at least the selected hard reserve;
- rank-skew wait falls by at least 25% in one primary regime;
- end-to-end throughput improves by at least 5% without a material tail-latency
  regression.

Stop or simplify if:

- the offline oracle misses its go gate;
- greedy is far from the oracle and an exact online method would exceed budget;
- replica selection shifts enough HBM work to Grace to erase the balance gain;
- reduced all-reduce residency does not reduce step wall;
- correctness requires an additional per-layer collective.

## Execution order

1. Capture step-level co-activation data and calibrate costs.
2. Build the offline oracle and decide whether the idea clears the go gate.
3. Probe real Grace allocation at the selected budgets.
4. Implement replica planning/loading with scheduling off.
5. Prove exactly-once correctness.
6. Add the minimal GPU greedy scheduler.
7. Run parallel c1/c4 A/B jobs and profile only the best candidate.
8. Update this directory with scripts, raw summaries, traces, and conclusions.

This ordering deliberately avoids a large loader/runtime change until the
captured-route oracle demonstrates recoverable end-to-end value.
