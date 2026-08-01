# Hot/cold Marlin tier balance

Status: **Stopped at the Phase 1 gate.** Rebalancing HBM slots across layers
gains nothing, because every layer is already cold-bound. The real finding is
that the cold tier runs at the C2C bandwidth limit and is 2.3x the hot tier,
so the only lever is more HBM residency, not a smarter split.

## The question

With the tight shared-memory policy from
[`2026-07-29-marlin-smem-monopoly`](../2026-07-29-marlin-smem-monopoly/README.md),
the hot and cold Marlin tiers run concurrently on two streams. Measured across
2,400 layer-instances of the Phase 5 traces, a layer's four Marlin launches
have a union of 310.0 µs against a sum of 441.8 µs — a ratio of 0.70.

The tiers are near-balanced in aggregate (hot 7.571 ms, cold 7.126 ms per
rank-step), so perfect overlap should approach 0.50. Where does 0.70 come from?

## The answer: overlap is already near-perfect; the tiers are imbalanced per layer

`analyze_tier_balance.py` separates the two candidate causes by comparing the
measured union against the floor the *observed chain lengths* permit. Even
flawless co-residency cannot beat the longer of the two tier chains.

| Per layer, `exact` arm | value |
| --- | ---: |
| Measured union | 310.0 µs |
| **Overlap floor** (max of the two chains) | **308.0 µs** |
| Left on the table by the kernels | **2.0 µs** (0.15 ms/rank-step) |
| Balanced floor (if both chains were equal) | 220.9 µs |
| **Recoverable by balancing the tiers** | **87.1 µs** (6.53 ms/rank-step) |
| Mean tier imbalance | 38.6% of the layer's Marlin work |
| Start skew between the streams | 1.2 µs |

**The kernels co-reside essentially perfectly: 310.0 against a floor of 308.0
is 99.4% efficient, and the two streams start within 1.2 µs of each other.**
Occupancy work — grid shape, register pressure, shared-memory footprint — has
at most 0.15 ms per rank-step in it and is not worth pursuing.

The entire gap is that within a layer one tier does ~39% more work than the
other, so the union is pinned to the longer one.

### It is variance across layers, not a global misallocation

Sampling 256 layers and identifying the hot tier as the one sharing the main
stream with the TP all-reduce:

- cold tier longer in **56.6%** of layers, hot tier longer in **43.4%**
- cold share of a layer's Marlin work: mean **0.529**, but **p10 0.263 and
  p90 0.768**

So the split is right on average and wrong almost everywhere individually.
This matters for the fix: the HBM budget does not need to grow, it needs to be
*redistributed across layers*. Layers whose cold tier dominates want more HBM
slots; layers whose hot tier dominates can give slots up. The total stays put.

## Why the existing placement does not already do this

The placement profile chooses which experts are HBM-resident to maximise hit
rate — how often a routed expert is found in HBM. That is the right objective
for a single-tier design and the wrong one here: two tiers running concurrently
are costed by `max(hot, cold)` per layer, not by the total. A layer can have an
excellent hit rate and still be 30% imbalanced, and it pays for that imbalance
on every step.

## Proposed lever

Choose the per-layer HBM slot count to minimise `sum over layers of
max(hot_time, cold_time)` under a fixed global slot budget, instead of
maximising aggregate hit rate. The machinery already exists:

- captured routes in `/e/scratch/profound/naeimitabiei1/claude-routing-profile-1047954-108`;
- the calibrated cost model — Grace is ~46 µs per distinct active cold expert
  and token-count independent, HBM is far cheaper and near flat (see
  `2026-07-31-replica-scheduling-v2` §3);
- `replay_exact.py`, which already replays layers under a cost model.

Note the interaction with replica assignment: that work balances cold experts
*across ranks*, this balances hot against cold *within a rank's layer*. They
are orthogonal — one moves work between ranks, the other between memory tiers —
but both change per-layer cold counts, so the placement optimiser must model
the replica assignment rather than assume primary-only ownership.

## Honest bound

6.53 ms/rank-step is 14% of the 46.1 ms GPU-busy step, but device time does not
convert one-for-one into step time: the replica work cut GPU busy 9.1% and
delivered 5–6% end to end. Scaling similarly, **expect 8–9% and treat anything
above that as a surprise**. A static per-layer allocation can only balance
*expected* durations, since routing varies per step, so some of the 38.6% is
irreducible.

## Phases

0. **Characterise** — done, above.
1. **Offline** — replay the captures, optimise per-layer slot allocation under
   the fixed budget, and report the modelled `sum of max(hot, cold)` against
   today's profile. Gate: ≥4% modelled reduction, else stop.
2. **Profile generation** — emit a v2 profile with the new per-layer hot sets,
   preserving owners and the replica map so the change is isolated.
3. **Measure** — same-node A/B, arms alternating, acceptance-free at batch 4
   plus MTP3 at batch 16, exactly as Phase 3 of the replica work.
4. **Confirm** — paired trace; the union/sum ratio should fall from 0.70 toward
   0.55 and per-layer imbalance from 38.6%.

## Artefacts

| File | What it is |
| --- | --- |
| `analyze_tier_balance.py` | Separates overlap efficiency from tier imbalance |
| `tier-balance-1167724.json` | Its output on the Phase 5 trace pair |


---

## Phase 1 result: the redistribution idea fails, for an instructive reason

**Gate: not met. Modelled gain from redistributing HBM slots across layers is
0.0%.** Per the plan's own stop criterion, this stops here.

### First, a correction to Phase 0

Phase 0 identified the hot and cold tiers by CUDA stream, reasoning that the
hot tier shares the main stream with the all-reduce. **That is wrong**: under
CUDA graph replay the stream ids rotate from layer to layer, so the labels were
close to random. The Phase 0 direction claim — "cold longer in 56.6% of layers,
hot in 43.4%" — is exactly what random labelling produces and should be
discarded.

The tiers are distinguishable by *grid*, which the trace records: the launch
policy gives the hot tier 2 CTAs per SM and the cold tier 1, so the hot tier
always launches 264 blocks and the cold tier 132. With correct labels a
regression of per-layer time against per-layer active expert count fits well
for the hot tier (R²=0.82) where the stream-labelled version fit nothing
(R²=0.006).

Phase 0's *quantitative* findings do not depend on the labels — union against
overlap floor, and the 38.6% imbalance — and stand unchanged.

### The corrected picture

| Per layer, per rank | hot tier | cold tier |
| --- | ---: | ---: |
| Active experts | 13.85 | 6.77 |
| Cost per expert | 9.75 µs | 45.32 µs |
| Chain time | 135.0 µs | **306.9 µs** |

The cold per-expert cost of 45.3 µs independently reproduces the 46 µs the
`2026-07-31-replicated-expert-scheduling` benchmark measured, from completely
different data. A cold expert costs **4.6x** a hot one.

So the cold tier is not occasionally the long pole — **it is 2.3x the hot tier
in essentially every layer**, with only modest spread (cold experts per layer:
mean 6.77, sd 1.25, range 3.36 to 9.04).

### Why redistribution cannot help

Moving an HBM slot from layer A to layer B makes A more cold-bound and B less.
That is only profitable when A has slack, meaning A is hot-bound. **No layer
is.** A greedy search over the fixed budget found zero profitable moves.

| Modelled Σ max(hot, cold) per rank-step | |
| --- | ---: |
| Today | 23.136 ms |
| HBM slots redistributed across layers | 23.136 ms (**+0.0%**) |
| Perfectly balanced tiers | 12.408 ms (−46.4%) |

The 46.4% is real but unreachable by reallocation: balance needs cold active
experts down from 6.77 to about 3.65 per layer, which means roughly 3.1 more
resident experts per layer per rank — about 4.7 GB more HBM against a measured
peak free of 3,874 MiB.

### The cold tier is at the hardware limit

6.77 experts x 20.05 MB in 306.9 µs is ~442 GB/s per rank, at or just past the
421 GB/s C2C roof measured in `2026-07-25-grace-bandwidth`. The cold tier is
not inefficient and cannot be tuned faster — it is moving weights as fast as
the link allows. The only way to shorten it is to move fewer weights, i.e. keep
more experts resident.

### What this means for the next lever

Not "balance the tiers" but "buy HBM residency". Candidates, none yet costed:

- **Reclaim HBM**: `gpu_memory_utilization` is 0.85 and the tiered reserve is
  7 GB. If ~5 GB can be freed safely, the modelled gain is large. This is the
  cheapest thing to test and should be next.
- **Shrink the resident copy**: the cold tier moves 20.05 MB per expert. Any
  reduction converts directly into cold-tier time at 4.6x the leverage of hot
  work.
- **Do not** pursue kernel or occupancy work on the overlap: Phase 0 bounded it
  at 0.15 ms/rank-step.
