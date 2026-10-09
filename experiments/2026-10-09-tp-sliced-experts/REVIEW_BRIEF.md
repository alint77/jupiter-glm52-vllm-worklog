# Review brief: TP-sliced tiered MoE decode kernel (GH200), 2026-10-09

## Goal
One persistent CUDA kernel (sm_90a, H100-class GPU of a GH200, 132 SMs) for the
MoE layer of GLM-5.3 (W4A16) at decode batch M = 8..32 tokens, TP4: each GPU
holds a 512-wide intermediate slice of EVERY expert (hidden 6144, expert inter
2048 -> slice 512, top-8 routing). Hot slices sit in HBM (~3.6 TB/s TMA
streaming measured); cold slices sit in this GPU's Grace LPDDR, read over its
own C2C link (419 GB/s measured, needs >= 48 CTAs with plain loads, ~11 GB/s per
CTA). Partial outputs are summed by the existing all-reduce. The bf16 shared
expert (its TP slice: w13 1024x6144, w2 6144x512) is fused into the same kernel
(it used to run on a side stream; a persistent kernel would block it).
Target: the memory roofline max(hot bytes / HBM, cold bytes / C2C) for M=8..32.
Constraints: same math/ops only (INT4 symmetric group-32 bf16 scales, fp16
activations with a per-row power-of-two scale, fp32 accumulate); no numeric
changes; no extra quantization.

## Design (td_v17.cu, the latest passing version)
- route_prep kernel (T+1 blocks x 1024 thr): placement/route lists into a global
  Workspace, fp16 x13 rows (+ bf16 x13b for the shared expert); PDL into:
- layer_kernel: grid 132 x (8 consumer warps + 1 producer warp), 1 CTA/SM,
  4-stage smem ring (stage 45 KB: 128x512 INT4 weights 32 KB + scales 4 KB +
  up to 8 token rows of activations), mbarrier full/empty.
- Work: "units" of 128 rows x 512 K (both w13 and w2 use this shape since v15).
  Per tier (hot / cold) a dynamic queue of groups claimed by one atomic each:
  S0 (shared w13, 12 chunks per group), R0 (routed w13, entry x 128-row tile,
  all 12 K chunks accumulate in registers), S1 (shared w2), R1 (routed w2, one
  unit per group, GR1 = 1). 16 "cold-preferring" CTAs start on the cold queue;
  a CTA whose queue is empty steals from the other tier.
- Producer warp: claims next group (prefetched one ahead), reads the entry
  record (ntok, local slot, tokens, routes, weights; for R1 also xs2[route]),
  writes per-stage desc + flush records (sd_row, sd_f) to smem, TMA 3D for
  weights + scales, cp.async.bulk for each token's activation row. For R1 it
  spins on ready[q][ei] == epoch (ld.acquire + fence.proxy.async) first.
- Consumers: per unit each warp covers a 64-row half x a 128-K quarter
  (4 k32 steps): LDS.128 of 4 row blocks, Marlin-style INT4 decode
  (1 SHF + 4 LOP3 + 4 HFMA2 per 32-bit word), mma.sync m16n8k16 f16 x2 per
  scale group, 4 FFMA to apply the bf16 group scale into fp32 acc.
  R1: after each unit, atomics (red.global.add.f32) of acc * (route weight *
  act scale) into y[token][HIDDEN]. R0: after the 12th chunk, reds into
  y13[route][2*INTER]; then per-thread fence, bar.sync, thread 0 atomicAdd on
  done13[q][ei]; the CTA whose add completes 96 units runs activate()
  (silu(g)*u -> fp16 x2 row + scale) and st.release ready[q][ei] = epoch.
- finalize kernel: y (fp32) -> bf16 out, zero y.

## Measurements (us per call, same GPU A/B; cells = hot/cold slices of entries)
| version | M=8 38/0 | 38/4 | M=32 110/12 | compute-only 38/4 | 110/12 |
| "today" (shipped kernel, slice layout, cuBLAS shared on side stream) | 102.7 | 115.5 | 323.6 | | |
| v15 | 87.0 | 94.4 | 265.9 | 71.0 | 190.6 |
| v17 | 84.0 | 90.5 | 250.1 | 65.0 | 171.1 |
Roofline share of v17: 0.73 (38/0), 0.68 (38/4), 0.67 (110/12). Floor at 38/4
~61 us. Compute-only = producer signals stages without any loads.
- route_prep takes ~4.4 us before the layer kernel passes its PDL wait; finalize
  ~1.5-3 us after; both are inside these numbers.
- CTA trace v17 38/4 (with loads): hot consumer warp 0 waits on full stages
  13.9 of 83.4 us; the producer is blocked on empty stages 20.8 us; producer
  spin on ready flags p50 3.4 us (hot), 7.2 us (cold CTAs). Cold CTAs: consumer
  waits 23.9 of 84 us; cold bytes ~19 MB in ~85 us (~220 GB/s, half of C2C).
- Microbenchmark probe_consume.cu: v15's consume block alone (no loads, no
  queue) = 0.72 us per 36 KB unit at 8 warps (0.77 with a bar.sync per unit),
  i.e. ~6.3 TB/s-equivalent over 132 SMs (1.8x the HBM rate). 16 warps: no
  faster. Non-volatile asm / hoisted LDS: no change.
- ncu (sm clocks locked by ncu), v15 compute-only: issue active 46%, tensor pipe
  5%, ALU 37%, FMA 30%; 160 regs; stall: wait 23%, selected 22%, not_selected
  10%, short_sb 8.5%, branch_resolving 8%, long_sb 8%. SASS sample shares
  (grouped by exec count): MMA blocks 48%, unit-loop head 14%, w2 per-unit
  flush 9%, w13 last-chunk flush 4% (fence.sc), kernel start 4%.
- ncu v17 WITH loads: MMA blocks 51%, unit-loop head ~12% (3.8% genuinely
  waiting on the full barrier), w13 count fence (per-thread fence.acq_rel ->
  MEMBAR.ALL.GPU + ERRBAR + CCTL.IVALL) 4.5% + barrier waiting on thread 0's
  fence+atomic 2.6%, kernel start (first full-barrier wait) 5.3%, w2 flush ~3%
  (ptxas turns the predicated red back into BSSY/BRA per element).

## History of what did / didn't help
v1 persistent kernel; v2 shared expert fused; v3 fast decode; v4 dynamic queues
+ stealing; v5 fused route_prep (slower); v6 ring addressed off smem (generic
LD -> LDS); v8 producer-written flush records; v9 16 consumer warps (slower);
v10 32-bit LDS / hoisted offsets; v11 conflict-free row-block mapping (4.3M bank
conflicts removed); v12 producer prefetches next claim; v13 torch-free builds;
v14 two 8-warp consumer groups (compute-only faster, overall slower: 96 regs +
spills, 2 stages per group); v15 128x512 w13 units (one consumer path);
v16 fused finalize (no gain); v17 lean loop head + predicated reds + acq_rel
fence (-4% / -6%).

## In flight (not yet validated)
- v18 = v17 + flush staged through smem ([token][68] floats per warp) and
  red.global.v4.f32.add over contiguous rows.
- v19 = v18 + the w13 done-count deferred: fence + count one unit later, the
  last-contributor check + activate one unit after that.
- One of {v18, v19, v19 without per-thread fence} just FAILED the per-slice
  check badly (51 fails, worst rel err 0.61); not yet known which.
- Planned v20: shared-expert w13 units statically assigned per CTA (they don't
  depend on routing) so their weight TMAs issue before griddepcontrol.wait,
  hiding the ~5 us pipeline fill behind route_prep.

## Open correctness issues (not root-caused)
- v12 with GR1 > 1 (several w2 units per claim): garbage in the first 64-row half
  of one w2 tile for a few tokens in ~10% of M=32 calls with cold experts, only
  when hot CTAs steal cold w2 work; smem weight checksums were correct.
- v14 with one consumer group fails the per-slice check (~0.011) rarely.
- Earlier "v13 rare failure without per-thread fence before the count" was seen
  with an older summed-slices check that had false positives (~6-7e-3).

## Questions for the reviewer
1. Is the design sound, or is there a structurally better decomposition for this
   regime (M=8..32, ~1-2 tokens per routed entry at M=8, 4.7 MB per entry)?
   e.g. different unit shape / split-K, 2 CTAs per SM, more warps with
   setmaxnreg, consumer-side software pipelining across units, cross-warp smem
   reduction before global reds, a different w13->w2 dependency scheme.
2. Is "consumer-bound by per-unit bookkeeping" the right diagnosis, given the
   probe (0.72 us/unit) vs ~1.3 us/unit in-kernel? What else could explain the
   gap, and the extra ~12 us consumer time with loads vs compute-only?
3. How to get C2C pegged (cold CTAs at ~220 GB/s of 419) and HBM saturated at
   the same time?
4. Memory-model correctness of: per-thread reds -> fence -> bar.sync -> thread 0
   fence + atomicAdd -> last one activates -> st.release ready; producer
   ld.acquire ready -> fence.proxy.async -> cp.async.bulk of the activation.
   Likely causes of the GR1>1 stealing race and of the v18/v19 failure.
5. What would you do next, in priority order?

## Update after the first reviews (Codex + fresh audit)
- Both found: v18 read sd_row[s] after releasing the stage (fixed in td_v21:
  each lane snapshots its 4 destination rows before the empty arrive); v19's
  deferred completion can deadlock (confirmed: both v19 builds hung the check;
  dropped).
- Cold-CTA sweep (v17, us, M=32): COLD_CTAS 12/16/20/24/32 at 110/12 =
  267.0/261.1/266.7/267.5/276.7; 96/8 = 219.5/220.5/224.0/224.5/235.0;
  124/16 = 319.9/305.0/304.7/308.5/321.5. 16 stays best: more dedicated cold
  CTAs do not help.
- GR1 sweep on cold-free cells (v17, M=8): GR1 1/2/4/6 at 30/0 =
  71.4/70.2/73.9/76.0, 38/0 = 84.2/84.2/85.7/86.6: amortizing the w2 claim +
  record does not help.
- v17 TD_NO_FLUSH_FENCE: passes 60 reps; same speed (84.3 vs 84.9, 93.2 vs 93.3).
- td_v20 (= v21 + w2 producer chain shortened): entry record in one unconditional
  read; xs13 for all tokens kept in producer lane registers (shuffle); x2 rows in
  the workspace carry their fp32 scale after the 512 halves (X2_LD = 520), so the
  consumer computes wt * xs2 from smem; the w2 weight TMA is issued before the
  ready spin, only the x2 bulk copies wait for it.
