# Decode MoE pair: per-expert pipelining of w13 -> act -> w2 (2026-10-04)

Plan and design for the next tiered-decode kernel change: replace the two
grid-wide barriers between `gemm_kernel<0>`, `act_kernel` and `gemm_kernel<1>`
with per-expert release flags, so w2's hot-expert work runs inside w13's cold
C2C drain and w2's cold weight reads start in the link-idle hot phase. Pure
scheduling change: same kernels, same tiles, same math, zero change to C2C
bytes. Source: `vllm/model_executor/layers/fused_moe/tiered_decode/tiered_decode.cu`.

## Measured basis

`trace-pdcp2` (Oct 3, serve.sh production config: DFlash2 k=7, DCP4, 400K,
3239/r2000 placement, fused AR+RMS on), 4 ranks x 27 decode steps at ~100K
context, vLLM-source completion traffic. MoE window (router->finalize, main
stream) **15.16 ms/step = 51% of the 29.57 ms step**. Per layer
(2,025 layer-instances, rank 0):

| component | p50 us | p10 | p90 |
|---|--:|--:|--:|
| router GEMM + topk + route_prep | 15.6 | | |
| w13 (`gemm_kernel<0>`) | 112.6 | 71.4 | 217.3 |
| act tail past w13 | 5.6 | 2.7 | 7.1 |
| w2 (`gemm_kernel<1>`) | 59.3 | 40.2 | 111.6 |
| finalize tail past w2 | 1.6 | 1.2 | 2.0 |
| window | 183.1 | 128.2 | 341.4 |

Launch structure is already optimal: w2 starts 0.3 us after w13 ends (0/2025
layers have >2 us), act and finalize are PDL-resident (their durations mirror
the GEMM windows; only the tails are exposed). There is no launch overhead
left; the window *is* the two GEMM durations joined by two grid-wide barriers.

CTA structure (TD_CTA_TRACE, 2026-10-04-decode-shared-overlap): hot CTAs done
~43 us (w13, ~2.4 TB/s HBM) / ~24 us (w2); cold CTAs at ~387 GB/s; w13 ends on
a cold CTA in 93-95% of launches; ~57% of SM capacity idles in the w13 window
after the hot tier retires.

**Caveat on traffic**: this trace draws 3.6 cold experts/GPU/layer (32%,
random-level, OOD). The live capture (job 2173771) says production Claude-Code
traffic draws 1.61-1.67 (15.5-16.8%). Both regimes are modeled below; the
arbiter is the step-0 replay, and a pipelined-agentic-traffic capture before
any "ship" claim.

## Cost model

Per cold expert, per rank, per layer (HIDDEN=6144, INTER=2048, int4 + scales):

| | weights | scales | total | link time @ 390 GB/s |
|---|--:|--:|--:|--:|
| w13 (4096x6144) | 12.58 MB | 1.57 MB | 14.15 MB | 36.3 us |
| w2 (6144x2048) | 6.29 MB | 0.79 MB | 7.08 MB | 18.2 us |
| total | | | **21.23 MB** | **54.5 us** |

(matches the 21,233,672-byte staging slot exactly; there are no wasted C2C
bytes anywhere in this path — cold reads are exactly the routed experts.)

Today the three resources idle as follows: the **link** is idle during w13's
hot phase (~15 MB of headroom = ~2 cold experts' w2 slices) and during w2's
hot phase; the **SMs** are 57% idle during w13's cold drain; and **w2's hot
work sits behind the barrier** instead of filling those SMs. Pipelining
converts the chain end from `w13_end + w2_len` into
`~max(link_total, hot_core) + one expert's tail`:

| cold c | chain today (us, core+glue) | pipelined bound | save/layer |
|--:|--:|--:|--:|
| 0 | ~85-95 | ~85-95 | 0-8 (per-expert w2-under-w13 only) |
| 1 | ~108-125 | ~100 | 10-20 |
| 2 | ~150-205 (measured (9,2) w13 alone is 127.5) | ~140 | 15-65 |
| 3 | ~200-240 | ~195 | 5-40 |
| >=4 | link-bound both ways | ~same | small |

The sweet spot is c = 1..3 — exactly where both real traffics live. Note the
measured (9,2) cell sits ~30 us above its own link roof, so schedule slack
(static role ranges, not link) is also on the table; per-expert release relaxes
that too.

## Design

### Current launch chain (what must change)

Five PDL launches per layer (`tiered_dep.cu` host side, ~1007-1060):
route_prep -> w13 -> act -> w2 -> finalize, each with
`programmaticStreamSerializationAllowed`. `griddepcontrol.wait` in a kernel
returns only at **full completion** of its stream predecessor, and
`launch_dependents` only permits the *launch* (prologue/residency) of the
successor:

* w13 PDL-waits on route_prep (lists), releases act at its own start
  (tiered_decode.cu:644, 670) — so act blocks are resident and spin during w13.
* act PDL-waits on w13 **completion** (:818), then does silu*up per route and
  releases w2 (:819).
* w2's producer warp already TMA-fetches its first `STAGES` weight stages
  before its PDL wait (:691-725), rows only after act completes (:727-733);
  consumers wait at :742.

So the only overlap that exists today is one ring (4 stages, ~215 KB) of w2
weights fetched during act's ~5.6 us tail. Everything else is grid-boundary
sequential; cold w13 and cold w2 C2C traffic never interleave.

### v1: chained per-expert release (two kernels, flags)

**Change 1 — act becomes per-expert.** Workspace gains per-expert unit
counters `done0[2][MAX_LIST]` and per-expert act-ready flags (route-level
counters suffice for act: `act_done[e]` over live routes). w13 CTAs, at the end
of each completed unit (t,c), `red.release.gpu.global.add` +1 to
`done0[tier][expert(t)]` (one lane; ~384 units/expert, ~5k atomics/layer —
noise). act drops its `pdl_wait`, keeps `pdl_release()` **at entry** (so w2
launches during w13), and each route block r acquire-polls
`done0[tier(r)][e(r)] == TILES<0>*CHUNKS<0>` before reading y13 rows. Hot
experts flip at ~43 us; their act rows are written immediately.

**Change 2 — w2 weights ungated, rows flag-gated.** w2's producer generalizes
the existing early-fetch pattern (:709-725) to *all* units: weight/scale TMA
for unit u issues as soon as a ring slot is free — no act dependency (w2
weights need only route_prep's list). The act x2 rows (per unit, :727-733) are
fetched only after acquire-polling `act_done[e(u)]`. Consumers drop the grid
`pdl_wait` (:742); a full mbar implies its rows were fetched post-flag.
Consequence: w2's cold weight reads start during **w13's hot phase**, when the
link is idle — this is the bulk of the win — and its hot expert chains run on
SMs retired from w13's hot tier, which is the same residency pattern the
aux-stream shared expert already exercises.

**Change 3 — order and hazards.**
* *Link ordering*: units stay in list order (hot CTA ranges then cold), so cold
  bytes arrive about expert-major; the link is work-conserving either way —
  total link time is fixed, only the last-byte arrival + per-expert tail
  matters. No bandwidth governor in v1.
* *Ring head-of-line*: a producer must not fill the ring with not-yet-ready
  units and starve a ready one. Weights are staged not-ready (bounded by
  ring capacity, 4 stages — self-limiting); the row fetch spin happens after
  the unit's weights are staged, and only for units whose turn it is. If the
  bench shows HOL stalls at high c, split expect_tx so rows join the same mbar
  at flag-flip (the :693-725 pattern already does exactly this for `pre`).
* *Reset discipline*: counters and flags are reset by their last reader
  (w2's last consuming CTA of e), the same scheme the file already uses for
  y13/y ("re-zeroed by their last reader", :106-107) — no host memset, graph
  replay-safe. Alternative if a corner race appears: monotonic counters plus a
  per-call generation tag.
* *Numerics*: unchanged tiles, unchanged accumulation order inside units, same
  silu*up math; only start times move. The exactness gates below must show it.

### v2 (only if v1 leaves bound on the table)

A single persistent kernel with cross-phase CTA reuse (w13 CTA picks up w2
units of the same expert). The v8 fully-fused attempt (dak kernel, MiMo) lost:
(9,0) 121-142 us vs 78 separate — per-flush fences made act's writes globally
visible at grid scope, 2-7 us act latency per tile, ring held only ~8 us of
prefetch. v1 avoids all three by keeping two kernels with per-expert (not
grid) release. Only pursue v2 if replay/bench shows >20% of the bound still
missing at c=1..3 cells.

## Expected speedup

| traffic | MoE window today | projected | save/step | step time |
|---|--:|--:|--:|--:|
| trace-pdcp2 (c~2.5-3.6, OOD) | 15.16 ms | 11.8-13.3 ms | 1.9-3.4 ms | 29.6 -> ~27 |
| agentic estimate (c~1.6) | ~11.5 ms (est) | 9.6-10.6 ms | 0.9-1.9 ms | ~26 -> 24-25 |

Knock-on, not promised: shorter and less skewed rank windows shrink the
post-MoE AR wait (2.02 ms/step here, up to 4.8 in the second sample) by some
fraction of the window saving; it is cross-rank max, so it scales with the
p90 layers pipelined. Throughput translation at unchanged acceptance: about
+4-8% decode tok/s on the agentic mix, more on cold-heavy steps. At c>=4 the
link bounds both forms and the gain vanishes — this is not a
high-cold-share rescue.

## Plan and gates

0. **Offline replay first (no GPU).** `replay_bound.py` over the existing
   TD_CTA_TRACE dumps (`fscratch/decode-cta/rank{0..3}-{start,stop}.pt`; CTA
   records are phase/block/SM/entry/ready/exit plus per-call hot/cold counts).
   Reconstruct per-expert chains under an ideal per-expert scheduler with the
   shared-link constraint; report the per-layer save distribution at this
   trace's mix, and (once captured) at the agentic mix. If the records lack
   expert identity, extend TD_REC with the unit's expert index and re-capture
   one window (the gpu_worker TD_TRACE_DIR dump hook is local-uncommitted —
   commit it as part of this step). **Go/no-go: modeled mean save >= ~0.8
   ms/step at the agentic mix; the c=0 cell must show no regression.**
1. **Isolated cell bench.** Standalone tiered_decode harness (dak bench
   pattern), pinned base clock, full Booster node: cells (hot x cold) over the
   routed distribution (h in 4..14, c in 0..4), chain vs pair-pipeline, plus
   tolerance vs the eager fp32 reference and the c=0 flag-overhead check.
2. **Integration**, behind `VLLM_TIERED_DECODE_PAIR_PIPELINE=1` (default off).
   Gates: greedy exact-text smoke, kernel census/tolerance tests, CUDA-graph
   capture (whole verify graph), exact-400K golden SHA (grid unchanged, but it
   is the project's rule after any Marlin-grid-adjacent change).
3. **Same-node serving A/B.** Three profiled windows per arm, same node,
   `step_breakdown.py` + collectives tables, agentic_bench tok/s, GSM8K-400
   and greedy deltas (expect no measurable change; report with CI anyway).
   Also capture one agentic-traffic TD_CTA_TRACE window and re-run the replay
   against reality.
4. **v2 decision** from the remaining gap to the replay bound.

## Out of scope

Placement/replica re-derivation from the live capture (separate thread,
data-dependent, orthogonal). Prefetch-to-HBM for decode (conditional on the
predictor hit-rate analysis on the capture; only pays when the link is the
binder). Any change to C2C bytes — there are none to save, and none added.
Anything touching the drafter/target math.

## Risks

* Producer HOL on the ring (mitigation in Change 3; detect in step 1 cells).
* Flag reset races (last-reader-reset is proven in this file; generation tags
  as fallback).
* w2 CTA residency competition with w13's tail cold CTAs (few CTAs; the aux
  stream already overlaps under the same 215 KB smem config).
* PDL gotcha: if act keeps releasing after its poll instead of at entry, w2
  never becomes resident and nothing changes — assert residency in the bench.
* Stale `torch_extensions/<name>/lock` if a bench server dies mid-JIT
  (known); FlashInfer ninja-deps race under concurrent servers (known).
* All numeric claims above are model + one trace; don't quote them as results
  until steps 0/3 run.

## Review (2026-10-04, second session)

The design work (launch chain, flags, last-reader reset, ring HOL, residency
assert) is sound. The speedup model is not, and two inputs are off.

**1. Overlapping w13 and w2 cannot beat the barrier chain beyond fixed
overhead.** Both tiers have the same 2:1 w13:w2 byte ratio (14.15 / 7.08 MB per
expert, hot from HBM or cold over C2C), so whichever tier binds w13 also binds
w2, by the same factor. With per-expert times hot `5.9 + 2.95 us` (~2.4 TB/s)
and cold `36.3 + 18.2 us` (~390 GB/s):

- today: `max(5.9h, 36.3c) + max(2.95h, 18.2c)`
- ideal pipeline: `max(8.85h, 54.5c)`

These are identical (both `1.5 * max(5.9h, 36.3c)`). Hot w2 running inside
w13's cold drain shortens nothing, because w2's binder is again the cold tier.
The premise that "the link is idle during w13's hot phase (~15 MB headroom)"
contradicts the CTA trace: cold CTAs stream from w13's first microsecond at
~387 GB/s and again throughout w2, so the link is busy for both windows. The
only idle link time inside the MoE window is the w13 -> w2 handoff (act tail
~5.6 us + w2's cold pipeline refill).

So the bound is **fixed per-layer overhead**: barrier + act tail, each
kernel's fill/drain latency, and the static hot/cold CTA split. Check against
the trace: at c ~ 3.6 the chain is ~142 + ~70 = 212 us vs a link-only 3.6 x 54.5
= 196 us, ~16 us of slack per layer. At the live mix (c ~ 2) expect ~5-10 us per
layer = **~0.4-0.75 ms/step**, not 0.9-3.4 ms. The "save/layer 15-65 us at c=2"
row follows from the idle-link premise.

**2. Corrections.**
- Live capture job 2173771 drew **2.00** cold per GPU per layer (19.3%), not
  1.61-1.67 (those are the agentic and older Claude Code held-out sets).
- `trace-pdcp2` was profiled (29.57 ms/step vs 24.5 ms unprofiled in the live
  `steps.csv`), and its decode ran at ~34K context (14K cached + 20K new), not
  ~100K. Absolute MoE-window numbers are inflated.
- Better data now exists: the live capture has exact step times (`steps.csv`,
  32K+ steps) and real routing; step 0 should model against it.

**3. What survives.** The measured (9,2) cell (w13 alone 127.5 us vs ~73 us
link-bound, ~53 us hot-bound) shows real schedule slack: static role ranges,
24 cold CTAs, fill latency. Per-expert release attacks part of that; a better
role split or work-stealing may attack it more directly.

**4. Proposed step 0 instead of the replay.** An analytic bound over the live
capture's per-rank, per-layer (h, c) distribution, using measured rates and the
CTA trace's per-kernel fill/tail latencies, reporting per step: (a) overlap
saving (expected: only the handoff gaps); (b) schedule slack vs each kernel's
roof; (c) the cross-rank max, since the post-MoE all-reduce waits for the
slowest GPU. If (a) is under the 0.8 ms gate, the targets are (b) and moving
cold reads into the link's idle time outside the MoE window (attention, dense
GEMMs, AR: ~half the step), i.e. predicted prefetch, which this plan lists as
out of scope but is the only place the link is actually idle.
