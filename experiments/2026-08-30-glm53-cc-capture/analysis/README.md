# Analysis: routing captured from real Claude Code usage

Snapshot `snap-1535650-a`, frozen at 151 records after 43 minutes of ordinary
Claude Code work against the capture host (job 1535650). 140 traces survive the
filters (11 dropped as shorter than 16 positions, **zero** dropped as identity
fallback), giving **120,344 routed positions** and 72.2M expert activations.

Reproduce with:

```bash
PYTHONPATH=$PWD .venv/bin/python \
  agent_space/experiments/2026-08-30-glm53-cc-capture/analysis/describe_routing.py \
  --trace-dir /e/fscratch/profound/$USER/caches/routes/snap-1535650-a \
  --profile shipped-2496=agent_space/profiles/glm53-w4a16-2496.json
```

## 1. Real turns are much longer than the synthetic driver's

| | Phase 44 driver | this capture |
| --- | ---: | ---: |
| traces | 389 | 140 |
| routed positions | 129,392 | 120,344 |
| mean positions/trace | 333 | **860** |
| median | — | 526 |
| p90 / max | — | 1,943 / 11,812 |

43 minutes of real use produced 93% of the entire synthetic corpus, from a
third as many requests. This is the distribution difference the phase was
looking for, and it shows up before any routing analysis.

## 2. GLM-5.3's router is broadly distributed

Share of routing mass captured by the top-k experts of 256, averaged over the
75 routed layers:

| k | 16 | 32 | 64 | 128 | 160 | 192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean share | 0.213 | 0.341 | 0.528 | 0.780 | 0.867 | 0.933 |
| worst layer | 0.098 | 0.182 | 0.336 | 0.601 | 0.719 | 0.826 |
| best layer | 0.298 | 0.432 | 0.618 | 0.855 | 0.923 | 0.970 |

Mean Gini is 0.412, ranging 0.145 (layer 3, nearly uniform) to 0.523 (layer 32).
**There is no free pruning**: only two layers contain an expert that was never
routed to, and each contains exactly one. Every other expert earns its place at
some point, so residency is a budget problem, not a dead-weight problem.

## 3. The shipped ranking is measurably wrong on real traffic

Cold-hit rate = fraction of routed expert activations that miss the resident
set. All figures on the 35-request held-out split, at the shipped budget of
9,984 hot experts (2,496 per rank).

| placement | held-out cold-hit |
| --- | ---: |
| even (no ranking) | 0.4749 |
| **shipped: Phase 44 synthetic ranking** | **0.2791** |
| **re-ranked on real usage** | **0.2103** |

Two readings, and the second is the one that matters:

- Re-ranking cuts cold-hit **24.6% relative**. The shipped ranking captures 74%
  of the gap between no ranking and a correct one; re-ranking takes the rest.
- The shipped ranking scored **0.2290 on its own synthetic held-out split** in
  Phase 44 and scores **0.2791 here** — it degrades 22% when moved onto real
  traffic. The synthetic driver was not measuring the same distribution.

Hot-set overlap is 0.796, so **20.4% of the resident set sits on experts real
usage does not want**. For comparison, Phase 44 replaced a GLM-5.2 placeholder
scoring 0.4047 and was worth +8.30% +/- 1.81% decode; this is a smaller move
from a better starting point, and its throughput value is unmeasured.

## 4. The error is uniform, not concentrated

Every one of the 75 routed layers improves under re-ranking; none is already
right. The ten worst layers carry only 21.4% of the total recoverable gap, so
there is no small patch — the ranking is wrong everywhere by a similar amount.

| | shipped | re-ranked | overlap |
| --- | ---: | ---: | ---: |
| mean over layers | 0.2791 | 0.2103 | 0.792 |
| range | 0.206-0.346 | 0.163-0.281 | 0.695-0.862 |

Worst: layers 36, 37, 33, 45, 55, 31, 34, 40 — the middle of the stack, each
losing 0.10-0.13 cold-hit. Best: layers 3, 4, 12, 13, losing about 0.03. The
early layers are both the least concentrated (Gini 0.145 at layer 3) and the
most generously provisioned (185-188 slots against a median of 127), which is
why a stale ranking hurts them least.

## 5. The capture has converged; more of it buys nothing

Ranking on N randomly drawn training requests and scoring on the fixed held-out
split, 5 seeds per point:

| train requests | routed positions | held-out cold-hit | stdev | hot-set overlap with full |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 14,984 | 0.2166 | 0.0116 | 0.924 |
| 25 | 16,559 | 0.2170 | 0.0041 | 0.929 |
| 40 | 30,895 | 0.2123 | 0.0046 | 0.952 |
| 55 | 41,068 | 0.2128 | 0.0037 | 0.968 |
| 70 | 53,849 | 0.2107 | 0.0013 | 0.969 |
| 85 | 64,300 | 0.2110 | 0.0013 | 0.983 |
| 105 | 78,892 | 0.2103 | — | 1.000 |

**Fifteen requests already recover 92% of the win.** Going from 15 to 105
requests moves held-out cold-hit by 0.006 absolute, under 3% relative, while
the seed-to-seed spread collapses from 0.0116 to 0.0013.

Three consequences:

- The live capture can stop. It has what it needs.
- The live-versus-replay question is moot for *this* ranking: replay would have
  bought a larger corpus that the curve says is not needed.
- A ranking is cheap to refresh. If the router's preferences drift with the kind
  of work being done, a fresh 20-request capture re-derives it in minutes.

## What this does not establish

Cold-hit is the objective the planner optimises, not throughput. Phase 44's
0.4047 -> 0.2290 was worth +8.30%; this is 0.2791 -> 0.2103 from an already-good
baseline, and the relationship between the two is not known to be linear. **The
number that decides whether to ship this is a paired A/B on the real-code decode
suite**, using `2026-08-29-glm53-w4a16/arm-quant-ab-2496.sh` with the profile as
the only variable, three pairs, because between-run spread on this configuration
is about 2.7%.

## 6. The profile, built

`finalize.sh` produced `results-snap-1535650-a/replicas-985.json`, installed as
`agent_space/profiles/glm53-w4a16-2496-realusage.json`. It validates against the
W4A16 checkpoint and matches the shipped profile's budget exactly -- 9,984 hot
experts, 3,940 replicas, same `config_sha256` and `index_sha256` -- so the
ranking is the only difference between the two.

Scored on the same 35-request held-out split (41,452 routed positions):

| profile | cold-hit | tail objective |
| --- | ---: | ---: |
| shipped, synthetic ranking | 0.2791 | 128.06 |
| real usage, frequency residency | 0.2098 | 104.08 |
| real usage, tail residency | 0.2098 | 103.98 |
| **real usage + replicas (shipped artefact)** | **0.2098** | **103.98** |

The tail objective improves 18.8%, more than cold-hit's 24.6% might suggest is
possible on the worst requests, and the three real-usage variants are within
0.1% of each other with hot-set overlap 0.999-1.000. **Residency strategy does
not matter here; the ranking does** -- the same conclusion Phase 44 reached, now
on real traffic. Overlap against the shipped profile is 0.795.

Replica placement was re-solved at the same 985 copies per rank, and the oracle
prices it at 1.205x: routed span 10.884 -> 10.044 ms at c1 (-0.840) and 26.504
-> 23.856 ms at c4 (-2.649).

Ship it only on the A/B. `submit-profile-ab.sh` runs three pairs, both arms in
one allocation, alternating which goes first.

## 7. What the planner optimises, and what it should

The planner's objective is the **cold critical path**: per token, per layer, the
max over EP ranks of that rank's cold expert count, summed over layers
(`_route_state` -> `cold_counts.max(axis=2).sum(axis=1)` -> `_tail_objective`).
Hot never enters it. It already takes a max across EP *ranks*; it does not take
one across *tiers*.

But the tiers overlap -- `_TIER_BLOCKS_PER_SM = {"hot": 2, "cold": 1}` splits SM
shared memory so hot and cold Marlin run concurrently -- so the real per-layer
cost is closer to `max(t_hot * H, t_cold * C)`. With the Phase 32 per-expert
costs of 9.75 us hot and 45.32 us cold, the two tiers balance at a **cold
fraction of 17.7%**. Below it, cold is hidden entirely and a cold expert is
free; above it, cold sets the step and hot is free.

`tier_balance.py` scores placements under that model, over the distinct experts
in a simulated step rather than per-token sums, since an expert activated by
several tokens in one step is staged and executed once.

### At today's operating point the two objectives agree

c4/MTP3, 300 sampled steps, held-out and training traces:

| | shipped | real usage | change |
| --- | ---: | ---: | ---: |
| planner objective (cold only) | 611.9 | 518.7 | **-15.2%** |
| max(hot, cold) step model | 27,748 us | 23,664 us | **-14.7%** |
| layers cold-bound | 75 / 75 | 74 / 75 | |
| mean cold/hot ratio | 2.42 | 1.93 | |
| hot work hidden under cold | 16 us/step | 156 us/step | |

The two agree to half a percentage point, because at 2496 slots **every layer is
cold-bound by roughly 2x**. While cold dominates, `max(hot, cold)` reduces to the
cold term and the planner's objective is a faithful proxy. The new profile is
not misdirected by the simpler objective.

### The objectives diverge sharply on the slot-budget question

**Withdrawn 2026-08-30.** The table below treats hot slots per rank as a
decision variable. It is not one in the shipped configuration:
`tiered_moe_planner.py:368` sets residency from available HBM and only honours
the profile's count when `VLLM_TIERED_MOE_PROFILE_CAP=1`, which nothing sets.
The sweep and the interior optimum near 3600 describe a knob the system does
not expose. Kept for the shape of the argument, which still holds if the count
is ever made binding.

| slots/rank | cold-only | max-model (c4) | cold/hot | cold-bound layers |
| ---: | ---: | ---: | ---: | ---: |
| 1200 | 1014.6 | 45,981 us | 6.33 | 75/75 |
| 2400 | 548.4 | 24,911 us | 2.08 | 75/75 |
| 2496 | 516.4 | 23,552 us | 1.92 | 74/75 |
| 3000 | 360.2 | 18,197 us | 1.22 | 61/75 |
| **3600** | 203.3 | **15,732 us** | 0.63 | 0/75 |
| 4200 | 78.7 | 16,118 us | 0.23 | 0/75 |
| 4800 | 0.0 | 16,476 us | 0.00 | 0/75 |

**The cold-only objective falls monotonically to zero; the max model has an
interior minimum near 3600 and rises after it.** That is the whole difference.
Past balance, converting a cold expert to hot adds 9.75 us to the tier that
*is* the critical path and removes 45.32 us from one that is already hidden --
strictly harmful. An optimiser told to minimise cold hits will happily spend
HBM to make the step slower, and will report that it succeeded.

### Where the model is wrong, and it matters

The measured sweep is 2400 -> 213.1, 2496 -> 214.8, **3000 -> 205.2 tok/s**. The
model gets 2400 -> 2496 right (-5.5% step time) but predicts 3000 should be
*better* still, and it measured worse. So the interior optimum is real but the
model does not locate it: something at 3000 costs throughput that
`max(t_hot*H, t_cold*C)` does not capture -- most likely HBM pressure against the
KV pin and the tiered reserve, which is a capacity effect rather than a
scheduling one.

Absolute times are also uncalibrated. The model puts the c1 routed span at 8,865
us where 2026-08-01 measured about 23 ms, though that measurement was at a much
lower residency (its own row here is between the 1200 and 1800 slot entries), so
the discrepancy is consistent rather than damning. **Only the ratio matters for
the balance point**, and the ratio is a direct measurement.

### What would actually change

1. **Objective**: `max(t_hot*H, t_cold*C)` per layer per rank, over the distinct
   experts in a step, replacing the per-token cold sum.
2. **Slot budget would become a decision variable** with an interior optimum,
   instead of "as many as fit" -- but only once `VLLM_TIERED_MOE_PROFILE_CAP=1`
   makes the count binding. Today it is not, and the Phase 46 non-monotonicity
   this pointed at has itself been withdrawn.
3. **Per-layer allocation stops being uniform in value.** A slot in a hot-bound
   layer is worth nothing; the current optimiser cannot see that and keeps
   feeding it. At 3000 slots, 14 of 75 layers have already crossed over.
4. **Replicas only pay on cold-bound ranks**, so the replica oracle inherits the
   same correction.

None of this changes the profile just built -- at 2496 slots the objectives
agree to 0.5pp. It changes what to do next, and it is the missing half of the
Phase 46 hot-slot question.
