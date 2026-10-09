# Audit: TP-sliced tiered MoE decode kernel (v17 to v19), 2026-10-09

This is a fresh-eyes review. I made no source edits and ran no GPU jobs. The new
numbers below come from re-analysing existing artifacts on the CPU:
`logs/tr-0cd10387.pt` (v17 trace, M=8 38/4), `tr-2abb4aa1.pt` (v15),
`tr-78d83409.pt` (v15 compute-only), `ncu-v17{ld,co}-sass.csv` (stall reasons per
region), `ab-v17*.jsonl`, and the `kdev roof` output. Appendix A has the
analysis script. `logs/codex-review.md` (16:16) independently reached the same
two v18/v19 bugs.

## TL;DR

1. **v18 bug (certain).** `flush_rows` reads `sd_row[s]` (td_v18.cu:811) after the
   warp has released stage `s` (`mbar_arrive_a(empty_s)` at :1186 and :1223). The
   producer is usually blocked on `empty` (20.8 us per call), so it rewrites
   `sd_row[s]` for the next unit at once. The last warps then add correct partials
   into the next unit's token rows, which explains a worst rel err of 0.61. v19
   inherits the bug, including the `TD_NO_FLUSH_FENCE` build.
2. **v19 bug (certain by construction).** It can deadlock. The deferred done13
   count (td_v19.cu:1230 and :1313-1315) runs only after the *next* full stage
   arrives. If this CTA's producer has meanwhile claimed an R1 group whose
   `ready` depends on that count, it spins forever before issuing that stage
   (:1047). `logs/r19.out` holds only one of the three check results. That is
   consistent with the two v19 builds hanging, but it is a hypothesis.
3. **The diagnosis is only half right.** The v17 trace splits the call into two
   regimes:
   - **w13 phase** (S0+R0+S1, about 48 us): consumer-bound at about 1.34 us/unit,
     with HBM at about 3.1-3.2 TB/s (about 88%).
   - **R1 phase** (the last about 35 us, 40% of the kernel): units issue every
     **2.14 us** (p10 1.86, p90 2.50) and consumers *wait* about 12 us. HBM runs
     at about **2.0 TB/s (57%)**. The consume work per unit is the same as R0's
     (ncu samples per unit agree within 3%), so the cause is the producer's
     serial chain of dependent global loads on 1-unit R1 groups, not
     bookkeeping. This is the largest single loss, about 10-16 us at 38/4.
4. **M=32 110/12 is probably bound by the cold tier.** C2C averages 257 GB/s over
   the whole 250 us. That is exactly 16 cold CTAs at about 16 GB/s each, and
   63.7 MB / 257 GB/s = 248 us. The cold/hot split was tuned at M=8 cells only.
5. **Fixed costs are about 9 us at M=8 (10%).** route_prep takes 4.9 us with HBM
   idle, and drain plus finalize take about 4.3 us. The shared expert (18.9 MB)
   does not depend on routing and can run under route_prep.

## 1. Correctness

### v18: stage metadata used after release (td_v18.cu)

- v17 copies `rw[]`/`fs[]` to registers at the loop head (td_v17.cu:1142-1145),
  before consuming and before `mbar_arrive_a(empty_s)`.
- v18 removes `rw` and reads `row = lds_u32(rows_u + c*4)` inside `flush_rows`
  (:811). That read is called after the arrive at :1186 (R0) and :1223 (R1).
- Once the 8th warp arrives, lane 0 of the producer passes `mbar_wait(&empty[s])`
  and lanes `< ntok` rewrite `sd_row[s][lane]` (the producer's R0/R1 branches).
- `ntok`, `fs` and `t` are still the old unit's values, read at the loop head.
  Only the destination row is wrong, so the errors are large and land in the
  wrong token rows. Weight checksums cannot see this.
- **Fix (same speed):** before the arrive, each lane loads the four rows it will
  need. In the store loop `k = lane + 32j`, `c = k>>4 = (lane>>4) + 2j` for
  j = 0..3, so 4 `lds_u32` per lane. Keep the arrive where it is. The invariant
  is: *a warp's empty arrival follows its last read of anything the stage owns,
  including `desc`, `sd_row` and `sd_f`.*
- Secondary issues:
  - The staging writes all 8 token columns even when `ntok` = 1. That is 16
    stores per thread with no predicate.
  - `red.global.v4.f32` still performs 4 scalar adds at L2. It cuts LSU
    instructions and addressing, not L2 atomic work. The bigger lever is to
    reduce the 4 K-quarter warps that hit the same rows (h1, k1 = 0..3) in smem
    first, which cuts the atomics 4x (section 3, D3).

### v19: deferred completion breaks forward progress (td_v19.cu)

The sequence:

1. Unit i is the last R0 unit of entry e, and this CTA's add would make
   done13 = 96. v19 sets `pn` (:1230) and `p1 = pn` (:1315).
2. `settle1` (the atomicAdd) runs only at the end of iteration i+1, after
   `mbar_wait_a(full)` for unit i+1 succeeds.
3. The producer's next claim is an R1 group of entry e. This is likely, because
   R1 groups are queued in entry order right after the R0 groups. It spins at
   :1047 before issuing any byte of that stage, and `ready[e]` can never be set.
4. Cross-CTA cycles of the same kind are also possible.

v17 is live because the count and activation run inside the iteration that
consumes the last R0 unit, and that stage was fully issued before the producer
moved on.

**Fix:** never block with a pending settlement. At the loop head, use one
non-blocking `try_wait`. If it fails while p1 or p2 is pending, settle first and
then wait. The better route is the warp-level completion in D4 below.

### v17 memory model: correct, but the fences are heavier than needed

- **Writer chain is valid.** The chain is: per-thread relaxed `red` → per-thread
  `fence.acq_rel.gpu` → `bar.sync 1` → thread 0 `__threadfence` + `atomicAdd` →
  `bar.sync` → last CTA `__threadfence` → `ld.cg` of y13. PTX fences are
  cumulative over operations ordered before them by `bar.sync`, and observation
  order runs through chains of atomic RMWs on done13.
- **The per-thread `fence.acq_rel.gpu` is probably unnecessary, and expensive.**
  It compiles to MEMBAR.ALL.GPU + ERRBAR + CCTL.IVALL, and its L1 invalidate is
  pure acquire overhead. CUTLASS's `GenericBarrier::arrive_inc` and split-K
  serial reduction rely on this pattern: every thread writes, then `bar.sync`,
  then thread 0 does `red.release.gpu`. That is the same protocol with one
  release instead of 256 fences.
- **The per-thread fence costs about 6% of the call with loads.**
  - Region 0a840-0a8d0 is 5.9% of samples with loads (78% `membar`), versus 3.3%
    compute-only. The fence waits for the thread's own red acks, and those slow
    down under memory load.
  - Region 0a9f0-0aad0 adds 1.6% (`barrier`).
  - Test: stress-check `td_v17:TD_NO_FLUSH_FENCE` alone, at least 100 reps x 16
    cases. Earlier "failures without the fence" came from the old summed-slices
    check, whose false positives were 6-7e-3. The v19-nofence failure is masked
    by the v18 bug.
- **Publishing x2 to the async proxy is double-fenced, and correct.** The writer
  does `fence.proxy.async` + `__threadfence` + `bar.sync` + `st.release`. Every
  copying lane does `ld.acquire` + `fence.proxy.async` before
  `cp.async.bulk`. xs2 is read with `ld.cg` after the acquire.
- **v17 liveness holds.** R0 and S0 groups never spin. The whole R0 or S0 queue
  section is claimed before any R1 or S1 claim. A spinning producer has already
  issued every stage of its earlier groups.
- **Other v17 shared state is safe.**
  - `s_last` reuse is safe: the next write by thread 0 comes after another
    `consumer_sync`.
  - `desc`/`sd_*` WAR across ring generations is ordered: consumer reads, then
    arrive (release.cta), then producer `try_wait` (acquire), then `__syncwarp`,
    then other lanes' `st.shared`.
  - `epoch`, `done13` and y/y13 reuse across calls are ordered by kernel
    boundaries and PDL. route_prep(N+1) cannot start before finalize(N)
    completes.
- **Stale comment.** td_v17.cu:757-758 says the w2 producer "issues the weights
  first and waits on ready only before loading the activation rows". The code
  spins before the weight TMA (:1009). That costs performance (section 3, D1),
  not correctness.

### Old open issues

- **GR1>1 with cold stealing.** I found no indexing, ready-caching or
  accumulator-reset bug. The flush records are read before release in v12 too.
  So it is not the v18 bug. "First 64-row half" means h1 = 0, which is warps
  0-3. Those are also the warps that run `activate` (warp < ntok). That points to
  timing skew, not data.
  - Cheapest decisive diagnostic: for a failing call, record per R1 unit
    `(epoch, q, ei, t, stage, it, CTA, fs[], rw[])` at the consumer *before*
    release. Then check whether the bad rows equal the expected value plus or
    minus exactly one unit's contribution: a missing, doubled or misrouted unit
    (bookkeeping), as opposed to noise (data or ordering).
  - Re-test on v17 first. Several things changed since v12, and GR1>1 has become
    a performance tool (D1).
- **v14 one-group failure** (0.007-0.008 in the README, "~0.011" in the brief).
  If 0.007-0.008 is right, it came from the old summed check and is likely a
  false positive. Re-run it on the per-slice check before chasing it.

## 2. Does "consumer-bound by per-unit bookkeeping" hold?

**Bytes per routed entry (slice):**

| Part | Size |
|---|--:|
| w13 weights | 3,145,728 B |
| w13 scales | 393,216 B |
| w2 weights | 1,572,864 B |
| w2 scales | 196,608 B |
| **Total per entry** | **5,308,416 B** (= 144 units x 36,864 B) |
| Shared expert, bf16 (= 576 units of 32 KB, about 3.6 routed entries) | 18,874,368 B |

The brief's "4.7 MB per entry" is weights only. The brief's "~19 MB cold in
85 us" also omits scales: it is 21.2 MB, so about 250 GB/s.

| cell | HBM bytes | HBM time at 3.6 TB/s | C2C bytes | C2C time at 419 GB/s | units | units/CTA |
|---|--:|--:|--:|--:|--:|--:|
| 38/4 | 220.6 MB | 61.3 us | 21.2 MB | 50.7 us | 6624 | 50.2 |
| 110/12 | 602.8 MB | 167.4 us | 63.7 MB | 152.0 us | 18144 | 137 |

The rate that keeps up with HBM is 1.05 us/unit with 132 streaming CTAs, or
1.19 us/unit with 116.

**The v17 38/4 trace, re-cut by phase** (my analysis; the per-R1-unit producer
records `td_record(90+q)` were already in the trace):

| span | time | units/hot CTA | us/unit | HBM achieved | consumer state |
|---|--:|--:|--:|--:|---|
| route_prep (HBM idle) | 0 → 4.9 | – | – | 0 | – |
| S0+R0+S1 ("w13 phase") | 4.9 → ~53 | 36 | ~1.34 | ~3.1-3.2 TB/s | busy; producer mostly blocked on empty |
| R1 phase | ~53 → ~85 | 15 | **2.14 issue gap** | **~2.0 TB/s** | **starved, waits ~12 us** |
| drain + exit + finalize | 85 → 92.3 | – | – | – | – |

The loss budget roughly matches the measured total:

| Item | Time |
|---|--:|
| Ideal HBM time | 61.3 us |
| route_prep | 4.9 us |
| Tail and finalize | 4.3 us |
| w13-phase shortfall | ~5.5 us |
| **R1-phase shortfall** | **~14-16 us** |
| **Sum** | **~92 us** (measured 92.3) |

What the measurements actually say:

- **The w13 phase matches the stated diagnosis.** It is consumer-bound at
  1.34 us/unit, about 13% above HBM rate. The excess is per-unit fixed costs:
  loop head, the count fence and barrier, and the start-up fill.
- **The R1 phase does not match it.** ncu samples per unit are about equal for
  R0 and R1 (with loads: (17.7+3.4)% / 2016 vs (33.1+7.9)% / 4032). So the
  consumer does the same work per R1 unit, yet units arrive 2.14 us apart.
- **The producer's serial chain per R1 group explains it** (td_v17.cu:978-1015).
  Each R1 group is 1 unit, and consecutive claims of one CTA are about 116
  groups apart, so each claim hits a new entry. The chain per unit is:
  1. claim atomic (prefetched, but the group's issue loop is short);
  2. `lists[q][ei]` (`ntok`, `local`);
  3. `tok`/`route` (predicated on `ntok`, so dependent);
  4. `ld.acquire ready` plus `fence.proxy.async`;
  5. `xs2[routel]`.

  That is 4-5 dependent L2 round trips at about 0.5 us under load, which gives
  about 2.1 us. Compute-only shows the same phase at about 1.5-1.6 us/unit
  (unloaded round trips of about 0.35-0.4 us). That is why "loaded vs
  compute-only" is +25 us end to end. The difference is about 12 us of R1-phase
  starvation, about 3 us of membar slowdown under load, and about 2 us of fill.
- **Caveat: the trace instrument sits on the path it measures.** In the trace
  build, `td_record(90+q, …)` (td_v17.cu:1052) adds an atomicAdd whose result
  the producer waits on, once per R1 unit. The un-instrumented R1 gap is probably
  1.6-2.0 us. The non-trace GR1 A/B below (E2) is the decisive test.
- **Why the probe and the kernel differ** (0.72 vs 1.1-1.2 us/unit even
  compute-only). The probe's loop is straight-line, so unit u+1's LDS overlaps
  unit u's HMMA/FFMA drain. In the kernel each unit boundary is a serial chain:
  `try_wait` → `BRA` → `LDS desc` → field decode → kind dispatch → `LDS` rows and
  scales → first weight `LDS`. Each R1 unit adds its flush and each R0 group adds
  its count. These are latency costs with 2 warps per scheduler, not instruction
  counts.
- **"16 warps no faster" in the probe is unexplained.** My pipe estimate per
  SMSP per unit is ALU about 55%, FMA about 30% and tensor about 15-35%. I see no
  obvious saturation. Running ncu on `probe_consume` would show it. That matters
  only if you go after occupancy, and I would not do that first.

## 3. Design critique (structural), with payoff estimates for M=8 38/4 and M=32 110/12

The overall family is right: persistent, warp-specialised TMA ring, dynamic
stream-K queues and a fused shared expert. The iteration missed three structural
issues (D1, D2, D5) and one sync-structure issue (D4).

**D1. R1 producer chain: −8 to −14 us at 38/4 (10-15%), −25 to −35 us at 110/12.**
Remove dependent round trips and pipeline the rest. All of these are the same
math:
- (a) Load the `Expert` record unconditionally, one coalesced word per lane
  (104 B), and shuffle it. That is 1 round trip instead of 2.
- (b) Put xs2 into the x2 row. `activate` writes the fp32 scale into the row's
  64 B pad, the bulk copy carries it (1040 B), and the consumer computes
  `fs = wt * xs2`. That is the same fp32 multiply, so it is bit-identical, and it
  removes the round trip after the acquire.
- (c) Issue the weight and scale TMAs *before* the ready acquire, as the old
  `TD_SPIN_EVERY_UNIT` layout did. Weights do not depend on readiness. This is
  live because R0 never spins.
- (d) Fetch the next group's record together with the prefetched claim (one
  group of lookahead).
- (e) Optionally use GR1 = 4 for the bulk of the R1 queue and 1-unit groups only
  for the last about 2 x 132 groups. One record and one acquire then serve 4
  units, but this needs the GR1 race settled.
- (f) Pick ready R1 work instead of claiming in entry order and spinning.
  Producer ready-spin is p50 3.4 us and max 16.4 us per hot CTA. On claim, test
  `ready` once. If it is not ready, push the group to a per-CTA deferred slot and
  claim another, or keep a ready queue filled by the activating CTA. Worth about
  1-3 us.

**D2. Cold/hot SM partition: −6 us at 38/4, −30 to −60 us at 110/12.**
Sixteen cold CTAs at about 16-20 GB/s each (TMA) cap C2C at about 260-330 GB/s,
and they take 12% of SMs away from the consumer-bound hot work.
- Cheapest fix: compute `cold_ctas` at runtime from bytes (the kernel already
  has `n_tier`). Balance `Bc/(n_c·r_c)` against `Bh/((132−n_c)·r_h)` with
  r_c ≈ 18 GB/s and r_h ≈ 27 GB/s per CTA. That gives about 16 at 38/4 and about
  20-24 at 110/12. As Codex noted, 4 cold entries give only 32 cold R0 groups,
  so at M=8 more cold CTAs cannot help w13 without splitting cold groups.
- Structural fix: **every CTA carries the cold tier.** Each CTA has one or two
  extra "cold slots" (a 5th stage fits: 231,552 B ≤ 232,448 B), filled several
  units ahead and consumed whenever their barrier completes. Then 132 x 36 KB of
  C2C is in flight (enough to saturate C2C) and hot work gets all 132 SMs.
- Alternative fix: hot producers issue `cp.async.bulk.prefetch.tensor.L2` for
  cold units a few microseconds ahead, so the cold TMA loads hit L2. This needs
  no smem but does need a deterministic prefetch order.
- Expected result: 38/4 approaches 38/0 (84.0 → about 78 after D1), and 110/12
  could drop to about 185-200 us.

**D3. R1 flush: −2 to −4 us at 38/4.**
The 4 K-quarter warps of a half (`k1` = 0..3) red-add the same 64 rows x ntok.
- Reduce them 4→1 in smem (scratch as in v18, but per half), then issue
  ntok x 16 `red.v4` from one warp, or one
  `cp.reduce.async.bulk.global.shared::cta.add.f32` of 256 B per token row.
- Summation order changes. The kernel header already states "numerics match
  Marlin's up to summation order", and fp32 atomics are already nondeterministic.
  Confirm that this counts as "same math" with the user before landing.
- The R0 flush (y13) has one owner per (entry, tile), as Codex noted. With a
  local reduction it could use plain stores, which also removes the y13 zeroing.

**D4. w13 completion without CTA-wide stalls: −3 to −5 us at 38/4.**
- Drop the per-thread fence (CUTLASS pattern).
- Make completion per warp-group instead of CTA-wide. Warps 1-7 do
  `bar.arrive 2, 256`, which does not block and still orders their prior
  accesses for the barrier's participants. Warp 0 does `bar.sync 2, 256`, then
  lane 0 does `atom.add.acq_rel.gpu`. If it was last, warp 0 alone runs
  `activate` (a loop over ≤ 8 routes) and `st.release ready`. The other 7 warps
  never stop. Warp 0's lag of 1-2 us is absorbed by the 3-stage slack.
- This is v19's intent without the liveness hazard. Apply the same to S0.

**D5. Fixed costs: −3 to −4 us at M=8 (about 4%), the same absolute at M=32.**
- (a) Move `prefetch.tensormap` above `pdl_wait` (td_v17.cu:934 vs :948).
- (b) Run shared-expert w13 under route_prep. Weights are immutable, so their
  TMA can issue before `griddepcontrol.wait`. The activation is x in another
  layout, and route_prep only permutes it.
  - The layer kernel can launch only after every route_prep block has executed
    `griddepcontrol.wait` and then `launch_dependents`. So the producer of x has
    completed by then, and x can be TMA'd in native row-major layout. The B
    fragment {x[g][2tq], x[g][2tq+1]} is contiguous, so it costs 2 x 32-bit LDS
    instead of one 64-bit LDS.
  - This is practically safe. Formally, PDL only promises visibility after the
    kernel's own wait, so the user should sign off.
  - The `done_s` reset and static S0 ownership must move out of route_prep, as
    Codex noted.
- (c) Use an evict-first L2 policy on the routed stream (TMA `.L2::cache_hint`),
  so the shared expert's 18.9 MB, prefetched to L2 during route_prep, survives
  until S1.
- (d) route_prep itself takes 4.4 us for a tiny kernel, which suggests a
  dependent-load chain. That is secondary to overlapping it.

**D6. Layout: +4% HBM rate, possibly much more over C2C.**
- Re-tile the slices so that each unit (32 KB weights + 4 KB scales) is
  contiguous: one 1D bulk copy per unit. The data and math are the same; only the
  storage order is new, and the TP-slice loader is new anyway.
- HBM: flat 3.91 vs w13-box 3.76 TB/s in your tma_probe. (The w2box at
  4.2-4.3 TB/s exceeds HBM peak, so that probe is L2-polluted: its pool is too
  small.)
- C2C: a cold w2 unit spans 1.5 MB (32 rows x 1 KB at a 48 KB stride). That is
  about 24 OS pages of 64 KB, on this `+64k` kernel with `cudaHostRegister`'d
  malloc memory.
- Cold units are visibly slower in the R1 phase (2.76 vs 1.79 us/unit for cold
  w13), which fits translation cost as well as the D1 chain.
- Also try `madvise(MADV_HUGEPAGE)` before `cudaHostRegister` in
  `GraceAllocation`.

**Not recommended first:**
- 2 CTAs/SM: 2 stages each is too shallow.
- 16 consumer warps or setmaxnreg: the probe shows no gain, and v14 spilled. If
  you revisit it, first drop `accs[2][4][4]` (32 registers held live for the
  shared expert even at M=8) by giving shared units a 16-register accumulator
  shape.
- Larger units: smem stages are already 45 KB.

**Bench fidelity notes:**
- The roofline uses 3.6 TB/s, but flat TMA reached 3.9. Against 3.9 the shares
  are about 8% lower.
- The hot pool is only "max cell + 16" experts. L2 reuse across replayed calls
  is small, but the shared expert is identical in every call.
- The C2C "≥48 CTAs" number is from plain loads. No TMA-over-C2C probe exists.

## 4. Next experiments, ranked by value per effort (E1-E3 need no new kernel code)

1. **E1. Cold-split sweep plus a TMA-over-C2C probe (about 1 h).**
   - `TD_COLD_CTAS ∈ {12,16,20,24,32}` at M=32 {96/8, 110/12, 124/16} and M=8
     {38/4, 46/6}.
   - Extend `tma_probe.py` with a Grace-registered source (flat, w13box, w2box)
     at {16, 24, 32, 48, 132} CTAs x {2, 4} stages, alone and concurrent with
     116/108 HBM CTAs.
   - **Confirms D2** if 110/12 drops by ≥ 10% at 20-24 cold CTAs and the probe
     shows w2box ≪ flat over C2C. **Kills it** if 110/12 is flat in the cold-CTA
     count.
2. **E2. Is the R1 phase producer-bound? (about 30 min)**
   - Run v17 with `TD_GR1 ∈ {1,2,4,6}` on cold-free cells (38/0, 110/0). No cold
     tier means no steal, so the known race cannot trigger.
   - Prediction if the chain binds: 38/0 goes from 84.0 to about 76-78, and the
     trace's R1 issue gap (Appendix A) falls from 2.1 to about 1.4 us.
   - If nothing moves, the R1 phase is memory- or consumer-bound, and D1 drops in
     priority.
3. **E3. Fence removal (about 30 min).** Run `td_v17:TD_NO_FLUSH_FENCE` through
   the per-slice check (at least 100 reps) and the A/B. Expect −2 to −4% and
   region 0a840 to fall from 5.9% to about 1%. Then fix v18 (row snapshot before
   release) and re-measure it on its own merits.
4. **E4. Build v20 as D1 a-d, plus D5a (about half a day).** Expect −10% at M=8
   38/4 and more at M=32. Confirm with the R1 issue gap of about 1.3-1.4 us,
   consumer full-wait under about 4 us, HBM in the R1 phase above 3 TB/s, and the
   per-slice check. Then add D4 (−3 to −5 us) with an explicit forward-progress
   argument.
5. **E5. All CTAs carry cold (D2 structural) or shared-under-route_prep (D5b),
   depending on E1.** If E1 shows cold binds at M=32, build the cold-slot design
   first (largest M=32 payoff). Otherwise do D5b (about −4 us flat).

Cumulative estimate if E2/E1 confirm: 38/4 goes from 90.5 to about 72-75 us
(roof share 0.82-0.85), and 110/12 from 250 to about 185-200 us (0.84-0.9).

## Appendix A: phase analysis of a trace (CPU only)

```python
# .venv/bin/python -I aud_tr.py logs/tr-<hash>.pt   (records as written by td_record)
import sys, numpy as np, torch
r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph, blk = r[:, 0] & 0xFF, (r[:, 0] >> 8) & 0xFF
t0 = r[ph == 50, 1].min(); us = lambda x: (x - t0) / 1e3
m = (ph == 90) | (ph == 91)            # one record per R1 unit issued by a producer
b, ts = blk[m], us(r[m, 3])
gaps = np.concatenate([np.diff(np.sort(ts[b == c])) for c in np.unique(b)])
print("R1 issue gap p10/p50/p90", np.percentile(gaps, [10, 50, 90]))
for p in (0, 1):                        # CTA record: {ready, last R0 flush, exit}
    sel = ph == p; cs = blk[sel]
    first = np.array([ts[b == c].min() for c in cs]); n = np.array([(b == c).sum() for c in cs])
    print(p, "firstR1 p50", np.median(first), "R1 us/unit p50", np.median((us(r[sel, 3]) - first) / n))
```

Note: the `hot entries 59, cold entries 58` header printed by `tl2.py` is wrong.
It takes the max of fields that the 90+q records reuse for `ntok` and `it`. The
cell really has 42 entries (ntok histogram 30x1, 6x2, 4x3, 2x5 = 64 routes).
