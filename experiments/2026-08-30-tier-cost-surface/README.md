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
