# Sliced hot set: one model-wide residency order (2026-10-18)

The user asked why one MoE layer's `layer_kernel` is far slower than its
neighbour in a 4 x 190K decode trace, whether the placement is per layer or
model-wide, and why the steps-touched profile (2026-10-09-expert-coupling,
touch3676) never reached prod. Then: rank the hot set model-wide by steps
touched, make the planner trim or fill by that ranking, measure, ship.

## Per-call MoE time is set by the cold experts the step touches

`route_record.py` in worktrees/pf5rt (pf5 = 144f2f4aee + a diagnostic: every
sliced decode MoE call writes its top-k ids into a GPU ring inside the CUDA
graph, `VLLM_ROUTE_RECORD=512`; the worker dumps it at each profiler
start/stop). Four arms at the c=4 config on the campaign holds
(`../2026-10-18-c4-swarm-timeline/launch_rt.sh`, `sliced-prof/rt-*`). Each
nsys window holds exactly as many verify steps as the ring recorded, so every
`layer_kernel` call is joined to its routes (`route_fit.py`, `an/calls-*.csv`):

| arm | M | fit, us per call | R2 per call | per-layer means |
|---|---|---|---|---|
| 4 x 100K / 190K (`ctx4`) | 16 | -7.9 + 1.33 h + 8.85 c | 0.82 | 0.84 |
| 4 x 5K / 50K (`dec4`) | 16 | -8.9 + 1.26 h + 9.46 c | 0.87 | 0.86 |
| 2 x 5K / 50K (`dec2`) | 8 | 4.6 + 1.31 h + 8.41 c | 0.78 | 0.84 |
| c=4 swarm replay (`tlnsys`) | 16 | -4.9 + 1.45 h + 6.23 c | 0.66 | 0.73 |

h, c = distinct hot / cold experts touched by the call (the server's own
maps). A cold expert costs ~5-7x a hot one. In the asked window (ctx4, MoE
31-34): cold 14.0 / 13.9 / 6.2 / 9.1 per step, 195 / 192 / 118 / 145 us,
corr(time, cold) over steps 0.99 / 0.99 / 0.68 / 0.95. MoE 0-5 touch 88.5 hot
/ 3.8 cold (spread routing, nearly all resident) vs 57.9 / 9.8 from MoE 20 on:
the same 155 us from different mixes.

## Why prod's hot set is off for the sliced layout

- Every profile so far ranks per EP owner GPU (3,676 owned experts each). The
  sliced planner (ep_size 1) loads the union of the four lists: an EP
  constraint the layout no longer has.
- The runtime count (14,290 at c=4) differs from the profile's 14,704; the
  planner trims one expert per layer round-robin, and the loader sorts each
  layer's list by expert id, so it drops the highest ids, not the least used.
- touch3676 won its served A/B under EP at c=1 (-0.117 +- 0.028 ms/step,
  8ec2406) but serve.sh was never switched; the sliced kernel shipped the next
  day.

## Fix

- `build_order.py`: one order over all 19,200 (layer, expert), by MTP3 verify
  windows (4 consecutive positions of a request) touching the expert, ties by
  route count; agentic train split + even live Claude Code captures.
  `profiles/glm53-w4a16-sliced-touch-order.json` (profile_version 3,
  `residency_order`).
- vLLM 65d9b56fb1: profile v3 carries `residency_order`; the planner takes the
  first owned entries that fit (no round-robin trim/fill). Tests in
  `test_tiered_moe_manifest.py` (the two `pinned_rounding_overflow` failures
  predate this: they fail on pf5 too).

Held-out score (agentic held-out + odd live captures; steps of n requests'
windows, distinct cold experts per step summed over layers):

| 14,290 hot (c=4) | M=4 | M=8 | M=16 | M=32 |
|---|--:|--:|--:|--:|
| prod as served (id-sorted lists, round-robin trim) | 188.8 | 364.8 | 691.0 | 1235.9 |
| prod lists in their own order, round-robin trim | 175.5 | 341.5 | 653.5 | 1185.7 |
| touch3676 (per GPU), round-robin | 166.6 | 324.5 | 624.0 | 1140.8 |
| model-wide route count | 164.8 | 321.5 | 619.8 | 1135.4 |
| **model-wide steps touched (shipped candidate)** | **164.5** | **320.6** | **617.7** | **1131.5** |

-10.6% cold per step at M=16 (-73 experts). At 12,500 / 16,000 hot: -147.6 /
-70.4 at M=16. With the fits' cold-minus-hot cost (4.8-8.2 us) that is ~-0.35
to -0.6 ms per M=16 step, before the served shrink seen before.

## Served A/B (running)

`launch_ab.sh`: ABBA on 8 holds (2392844-45, 2393362-67), a = pf5 + ccfreq3676,
b = pf6 (65d9b56fb1) + the new profile, fresh compile caches; two holds per
kind x order: c=4 swarm replay with nsys windows, dec1 (M=4), dec4 (M=16).

Round one (unpaired: a and b on different holds; verify-step medians, ms;
round two stalled 25 min on node_run2.sh's `wait` for its own mem_sampler,
released by hand):

| arm | M | span a | span b | MoE a | MoE b |
|---|--:|--:|--:|--:|--:|
| dec1 ~5K | 4 | 14.31 | 14.25 | 4.69 | 4.63 |
| dec1 ~50K | 4 | 15.80 | 15.41 | 5.89 | 5.49 |
| dec4 ~5K | 16 | 20.59 | 20.38 | 9.41 | 9.40 |
| dec4 ~50K | 16 | 23.68 | 23.25 | 12.16 | 11.85 |
| swarm (2 holds each) | 16 | 23.21 / 22.83 | 26.40 / 23.25 | 10.98 / 10.66 | 14.56 / 11.09 |

The swarm b on 2392845 (+3.5 ms MoE) is beyond what the cold-count change can
move (the fits put a whole -10% at ~-0.5 ms); round two runs a on that hold.

Paired (both rounds, b - a on the same hold, `ab_compare.py` ->
`an/ab_compare.txt`; verify-step medians, ms):

| arm | M | span b-a per hold | MoE b-a per hold | mean span / MoE |
|---|--:|---|---|---|
| dec1 ~5K | 4 | -0.14, -0.16 | -0.06, -0.04 | -0.15 / -0.05 |
| dec1 ~50K | 4 | -0.20, +0.58 | -0.13, +0.74 | +0.19 / +0.30 |
| dec4 ~5K | 16 | -0.43, -0.43 | -0.23, -0.38 | -0.43 / -0.30 |
| dec4 ~50K | 16 | +0.20, -1.11 | +0.21, -1.11 | -0.46 / -0.45 |
| c=4 swarm | 16 | +0.04, -0.25, -0.09, -0.33 | +0.08, -0.34, -0.00, -0.36 | -0.16 / -0.16 |

Hold 2392845 is slow for both arms (a 26.65 / 14.90 ms span / MoE vs ~23 /
~11 on the others): the round-one swarm outlier was the node; b - a there is
-0.34 MoE. MoE b <= a in 9 of 12 pairs, mean -0.14 ms/step; the ~50K dec
windows swing +-1 ms between holds (few, long steps), the rest agree in sign.

Shipped before round two on the user's call (the order ranks by the quantity
the per-call fit says sets MoE time, it wins held out at every M and budget,
and it removes the id-sorted trim): `claude-glm53-sliced-mtp3.sh` sets
`PROFILE=glm53-w4a16-sliced-touch-order.json`. serve.sh's default stays the
EP replica profile (the EP launcher needs its replicas).
