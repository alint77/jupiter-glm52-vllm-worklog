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
