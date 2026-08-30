# The tiered-MoE cost surface: is `max(t_hot·H, t_cold·C)` even right?

## Why

The placement optimiser prices a layer as **linear in cold experts** and ignores
hot entirely. `2026-08-30-glm53-cc-capture/analysis` extended that to
`max(t_hot·H, t_cold·C)` using Phase 32's 9.75 µs and 45.32 µs per expert, and
concluded there is an interior optimum in the hot-slot budget near 3600.

Both constants come from a **regression against per-layer active-expert counts
in one production trace** (hot fit R²=0.82), so they are local slopes at a single
operating point — hot 13.85, cold 6.77 experts per layer per rank. Using them
across a sweep from 1200 to 4800 slots, where hot ranges roughly 5 to 21 experts
per layer per rank, extrapolates far outside their calibration range.

Three mechanisms say the surface is not a plane:

1. **The tiers are not independent.** The tight shared-memory policy that lets
   them overlap (`_TIER_BLOCKS_PER_SM = {"hot": 2, "cold": 1}`) makes each tier's
   kernel slower in isolation than the default launch would. A tier's cost is
   therefore a function of whether the other is co-resident, which a
   `max(a, b)` of two independently measured times cannot express.
2. **Marlin changes regime with token count.** It is weight-bandwidth-bound at
   small m and moves toward compute-bound as m grows. The hot tier feels that;
   the C2C-bound cold tier largely does not. So `t_hot / t_cold` — and with it
   the balance point — is a function of m, not a constant. MTP3 at c1 gives
   m = 4 and at c4 gives m = 16.
3. **Both tiers launch a fixed grid** (SMs × 2 hot, SMs × 1 cold) regardless of
   how many experts they execute, so per-expert cost is quantised by how the
   expert count divides into that grid. **(4 hot, 1 cold) and (8 hot, 2 cold)
   share a ratio but need not share a per-expert cost.**

There is also a hard regime boundary the model ignores: `_apply_tier_launch_policy`
sets `max_tokens = (num_speculative_tokens + 1) × max_num_seqs`, which is **16**
in production. At or below that the tiers overlap; above it the policy is
dropped and they run serially. c4/MTP3 sits exactly on the boundary.

## What is measured

`agent_space/benchmarks/tier_cost_surface.py` sweeps (H, C, m) and records three
times per point, all under **CUDA graph replay**:

| | |
| --- | --- |
| `hot_us` | hot tier alone, 2 CTAs/SM, tight smem |
| `cold_us` | cold tier alone, 1 CTA/SM, tight smem, weights in NUMA-local pinned Grace |
| `union_us` | both, forked onto two streams and joined, as production launches them |

Graph replay is not optional here: timing a fork/join eagerly charges two stream
barriers per iteration, a ~110 µs floor on Booster that would swamp the effect.

Weights use the deployed GLM-5.3 W4A16 **group 32** layout, 21,233,664 bytes per
expert, not the group 128 the older `tiered_marlin_dispatch.py` benchmark used.

## Hypotheses

Stated before the run so the result can falsify them.

- **H1** `hot_us / H` is not constant in H — a staircase from grid quantisation
  rather than a line.
- **H2** `t_hot / t_cold` differs between m = 4 and m = 16, moving the balance
  point with concurrency.
- **H3** `union_us > max(hot_us, cold_us)`: overlap is imperfect, so the max
  model is optimistic and the true cost sits between max and serial.
- **H4** Scale invariance fails: (4, 1) and (8, 2) have different per-expert
  costs despite the same ratio.

If H1–H4 all hold, the placement objective needs a **measured lookup surface**
rather than two constants, and the interior optimum near 3600 slots is not
trustworthy as stated.

## Method notes

- Booster only. The login node prefers a different Marlin grid (66 CTAs against
  132) and its C2C behaviour differs, so a surface fitted there describes the
  wrong machine.
- `numactl --cpunodebind --membind` is load-bearing:
  `GraceAllocation.allocate_pinned` binds nothing and requires the caller to be
  bound already. Production gets that from `--numa-bind`; a bare `sbatch` does
  not, and the cold weights would land at 0% locality.

## Status

- 2026-08-30: submitted as job 1535786.

## Result: the threshold is wrong, and there is no regime where it is right

`boundary-decode-1535798.json`, at the decode-realistic split Phase 32 measured
(13.85 hot / 6.77 cold active per layer per rank, rounded to 14 / 7):

| tokens | policy on | policy off (today, above 16) | gain |
| ---: | ---: | ---: | ---: |
| 4 | 386.8 | 463.8 | +16.6% |
| 8 | 387.7 | 463.8 | +16.4% |
| 16 | 387.7 | 468.2 | +17.2% |
| 24 | 741.9 | 897.0 | **+17.3%** |
| 32 | 743.9 | 896.4 | +17.0% |
| 64 | 1096.0 | 1374.3 | +20.2% |
| 256 | 2519.3 | 3148.4 | +20.0% |
| 1024 | 3902.2 | 4994.8 | +21.9% |
| 4096 | 12110.8 | 15438.6 | +21.6% |
| **8192** | **23877.9** | **30199.1** | **+20.9%** |

**The overlap policy wins at every token count from 4 to 8192, by 16-22%.**
There is no crossover. The comment's claim that above the threshold "the default
heuristic's grid is the right one" does not hold anywhere it was checked, and
8192 is exactly `max_num_batched_tokens`, the prefill chunk size.

At the prefill-realistic split (33 hot / 31 cold) the same shape holds, +8.9% to
+24.3% for m >= 8.

### A retraction

The first run reported **-24.8% at m=4**, i.e. the policy hurting. That was an
artifact of this benchmark, not a finding: 4 tokens x 8 = 32 routing slots
cycling over 64 targets only reach the first 32, which were all hot, so the cold
tier had no work and the policy only slowed the hot kernel down. It does put a
number on the cost of the tight-smem launch when there is nothing to overlap
with -- about 25% on the hot tier alone -- but it says nothing about the
threshold. The decode-realistic rerun above is the valid measurement.

### The cost surface run is confounded; discard it

`surface-1535795.json` conflates two variables. As H grows at fixed C, the
tokens routed to each cold expert fall, because the benchmark cycles `m * TOPK`
slots over `H + C` targets. Cold time at C=4 reads 433.9 us when H=8 and 231.8
us when H=16 -- that is tokens-per-expert moving, not cold expert count. A
redesign must hold tokens-per-expert fixed.

Two things it does show cleanly:

- **H1 holds, and more strongly than "staircase".** Hot cost is not monotonic in
  H: 8 -> 148.4 us, 12 -> 207.9, 16 -> **158.0**, 20 -> 191.7, 24 -> 220.8.
  Twelve experts cost more than sixteen. There is no per-expert cost.
- **H3 holds and is occasionally severe.** Union is usually within ~4% of
  `max(hot, cold)`, but H=8/C=2 gives 376.9 against a max of 231.2 (+63%) and
  H=24/C=8 gives 651.7 against 436.1 (+49%) -- worst exactly where the tiers are
  closest in size, which is the regime the balance-point argument lives in. The
  interior optimum near 3600 slots that `tier_balance.py` reported should not be
  trusted as a number.

## What to change, and the two cautions

Raising `tiered_overlap_max_tokens` is worth 17-21% of routed-MoE time in every
step above 16 tokens, which is every prefill and every mixed prefill-decode step.

1. **The threshold gates two unrelated things.** Besides the launch policy,
   `prepare_replica_routing` returns False above it, so raising one value would
   silently enable replica assignment at token counts where it has never been
   exercised. Replica assignment is only safe when every EP rank derives the same
   routes. **Separate the two values before raising either.**
2. **This is a single-layer microbenchmark.** The routed MoE is a fraction of
   prefill wall clock, so 20.9% here is not 20.9% end to end. It needs a paired
   server A/B, and the exact-400K golden SHA re-established, which the worklog
   requires after any Marlin grid change.
