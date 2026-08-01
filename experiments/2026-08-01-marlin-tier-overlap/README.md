# Hot/cold Marlin tier balance

Status: **Phase 0 done — the cause is identified and it is not what it looked
like.** The lever is worth ~6.5 ms per rank-step, larger than the replica
scheduling in [`2026-07-31-replica-scheduling-v2`](../2026-07-31-replica-scheduling-v2/README.md).

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
