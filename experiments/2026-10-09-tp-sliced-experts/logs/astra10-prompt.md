# Astra consult 10: the sliced MoE layer_kernel's producer warp is too slow per w2 unit at M=8

Do NOT run any commands (your sandbox cannot run on this login node). Reason from the material below.

## Context
Persistent Hopper (H100/GH200, sm_90a) W4A16 MoE decode kernel, `layer_kernel`, GRID=132 CTAs, THREADS=288:
8 consumer warps (INT4 decode + mma.sync m16n8k16) and 1 producer warp (warp 8) that feeds a 4-stage
TMA ring (46 KB / stage: 32 KB weight tile + 4 KB scales + up to 8 activation rows of 1088 B).
Work = queues per tier (hot q=0 / cold q=1) of groups claimed by atomicAdd on `ws->next[q]`:
S0 (shared expert w13), R0 (routed w13, 12 chunks per group), S1, R1 (routed w2, GR1 tiles per group;
g1 = T > 8 ? 2 : 1). w2 units of an entry wait on `ws->ready[q][ei]` (released by whichever CTA finishes
the entry's last w13 chunk; it runs the activation and st.release's the flag). Consumers go stage by
stage in ring order; the whole CTA's 8 warps arrive on `empty[s]`.
Rules: same math only; only weight TMAs may be issued before the PDL wait; no calibration-based balancing.

## Measurements (M=8 tokens, 38 hot experts / 4 cold, untraced kernel ~91 us; floor 61 us)
Per hot CTA: ~31 R0 units (w13) then ~15 R1 units (w2), GR1=1. Consumer per R1 unit ~1.4 us
(0.93 math + 0.22 flush + 0.22 loop head). From ~56 us on, hot CTAs wait 30-40% of the time,
~8-12 us / CTA on w2 loads that were issued before the consumer arrived but land after it.

Producer per R1 unit (globaltimer stamps, 32 ns tick, values in us, mean):

| step | us |
|---|--:|
| claim atomic issued (lane 0) | 0.06 |
| group_at + entry index math (runtime division by TILES1/g1) | 0.22 |
| entry record load (5 LDG from ws->lists[q][ei], then shfl of xs13 by token) | 0.32 |
| record -> empty-wait start (sd_row/sd_f STS, loop setup) | 0.32 |
| empty wait | 0.17 |
| desc STS + expect_tx + 2 weight TMAs | 0.44 |
| ready spin: ld.acquire.gpu (LDG.STRONG + CCTL.IVALL), taken every unit (each claim is a new entry) | 0.22-0.30 |
| x2 row bulk copies (lanes < ntok) | 0.22 |
| claim shfl + loop head | 0.16 |
| **sum** | **~2.2** |

So the producer issues one w2 unit per ~2.2 us vs the consumer's 1.4 and the ring drains in the w2 phase.

Things that did NOT help, and why (measured):
- v40: claim two groups ahead + prefetch the next group's entry record into registers during the
  current group's issue. Slower (91.2 -> 94.0). SASS: the claim ATOMG and the record LDGs share
  scoreboard SB5; the first use of the (prefetched) record waits on SB5 and so waits for this group's
  contended claim atomic (record wait 0.32 -> 0.65 us).
- v41: `len[q]` / `n_tier[q]` were a 16 B stack frame (LDL per group); made register-resident. Neutral.
- v42 ablation: static strided schedule (no claim atomics at all, no stealing): the setup still
  costs 0.79 us/unit and overall the kernel is far slower (no stealing -> imbalance).
- GR1=2 (2 tiles per claim, shares claim/record/ready): helps M=16/32 (145 vs 152, 242 vs 255 us)
  but neutral at M=8 (tail imbalance); guided self-scheduling (claim size shrinking) lost to static GR1=2.

## Options I am weighing
A. Two producer warps (THREADS 320; regs 161/thread fit). Each claims its own groups and reserves ring
   positions with an atomicAdd on a shared-memory counter (reserve nch consecutive positions per group),
   so consumers still see one stage sequence. Each producer waits empty[pos % 4] with phase from pos.
   End: each producer reserves one position and writes K_END; consumers keep going until they have
   seen 2 ENDs (the first END slot is just released). Per-producer state (q, stolen, last_ready) is
   private. The R0 inner loop (consumers consume a group's 12 chunks in consecutive stages without
   re-reading the descriptor) still works since a group's positions are consecutive.
   Concern: a producer that reserved positions far ahead (behind the other's 12-chunk R0 group) holds
   a claim it cannot issue yet; the claim order no longer matches issue order.
B. Shorten the chain in one warp: template the producer on g1 (no runtime division); decode R1 groups
   incrementally; issue the 2 weight TMAs BEFORE the record-dependent STS (the weights need only
   `local`, t); hoist the ready poll; use `ld.relaxed` + one fence for the ready poll instead of
   ld.acquire per poll (CCTL.IVALL); move the claim atomic so nothing waits on its scoreboard
   (e.g. issue it from a different warp, or after the last SB-waiting instruction).
C. A tiny "scheduler" warp that only claims groups and loads records into an smem queue
   (mbarrier-signalled), the producer reads them from smem.
D. Something else.

## Producer code (td_v41.cu, lines 1163-1445)
```cpp
  if (warp == CONSUMER_WARPS) {  // producer warp
    if (lane == 0) {
      for (int q = 0; q < 2; ++q)
        for (int k = 0; k < 2; ++k) {
          asm volatile("prefetch.tensormap [%0];" ::"l"(
                           reinterpret_cast<uint64_t>(&p.tier[q].w[k]))
                       : "memory");
          asm volatile("prefetch.tensormap [%0];" ::"l"(
                           reinterpret_cast<uint64_t>(&p.tier[q].s[k]))
                       : "memory");
        }
      if (sh)
        for (int k = 0; k < 2; ++k)
          asm volatile("prefetch.tensormap [%0];" ::"l"(
                           reinterpret_cast<uint64_t>(&p.sw[k]))
                       : "memory");
    }
    TD_V(const uint64_t td_p0 = td_now(); uint64_t td_spin = 0, td_empty = 0;)
#ifdef TD_UNIT_TRACE
    unsigned td_ps = 0;  // producer lane 0: this CTA's producer records
    if (lane == 0) td_ps = atomicAdd(&td_trace_n, 512u);
#endif
    const float xs13r =
        lane < T ? ws->xs13[lane] : 0.f;  // token lane's x13 scale
    int q = own, last_ready = -1, it = 0;
    bool stolen = false;
    int gi = 0, gk = 1;  // the current claim and its unit count
#ifdef TD_RECPF
    // the next claim, and the entry record prefetched for it
    int gq = 0, pf_e = -1, pf_q = -1, pf_ntok = 0, pf_local = 0, pf_tok = 0,
        pf_route = 0;
    float pf_wt = 0.f;
    if (lane == 0) {
      gi = atomicAdd(&ws->next[q], 1);
      gq = atomicAdd(&ws->next[q], 1);
    }
    gq = __shfl_sync(0xffffffffu, gq, 0);
#else
    if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
#endif
    gi = __shfl_sync(0xffffffffu, gi, 0);
    while (true) {
      if (gi >= (q ? len[1] : len[0])) {
#ifdef TD_NO_STEAL
        break;
#endif
#ifdef TD_NO_STEAL_COLD
        if (q == 0) break;  // hot CTAs never take cold work
#endif
        if (stolen) break;
        stolen = true;
        q ^= 1;
        last_ready = -1;
        gk = 1;
#ifdef TD_RECPF
        if (lane == 0) {
          gi = atomicAdd(&ws->next[q], 1);
          gq = atomicAdd(&ws->next[q], 1);
        }
        gq = __shfl_sync(0xffffffffu, gq, 0);
#else
        if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
#endif
        gi = __shfl_sync(0xffffffffu, gi, 0);
        continue;
      }
#ifdef TD_UNIT_TRACE
      if (lane == 0) td_record_at(td_ps++, 170, td_now(), 0, 0, 0, 0);
#endif
      // claim the next group now: its round trip overlaps this group's issue
      int gn = 0, kn = 1;
#if TD_GUIDED > 0
      {
        // past the start of R1 the counter only moves through R1 units
        const int r1s = (q ? len[1] : len[0]) - (q ? n_tier[1] : n_tier[0]) * TILES1;
        if (gi >= r1s)
          kn = max(1, min(TD_GUIDED, ((q ? len[1] : len[0]) - gi - gk) / (2 * GRID)));
      }
#endif
#ifndef TD_NO_PREFETCH_CLAIM
      if (lane == 0) gn = atomicAdd(&ws->next[q], kn);
#endif
      Group gr = group_at(gi, (q ? n_tier[1] : n_tier[0]), q == 0 && sh, g1);
#if TD_GUIDED > 0
      if (gr.kind == K_R1) gr.nch = min(gk, (q ? len[1] : len[0]) - gi);
#endif
      const Tier& tr = p.tier[q];
      // a routed group's entry record, once per group
      const bool routed = gr.kind == K_R0 || gr.kind == K_R1;
      int ei = 0, t0 = 0, ntok = 0, local = 0, tokl = 0, routel = 0;
      float fl = 0.f;
      if (gr.kind == K_R0) {
        ei = gr.x / TILES0;
        t0 = gr.x - ei * TILES0;
      } else if (gr.kind == K_R1) {
        ei = gr.x / (TILES1 / g1);
        t0 = (gr.x - ei * (TILES1 / g1)) * g1;
      }
      const auto load_record = [&](int e_) {
        // one round trip: every field at once (lanes past ntok read a valid
        // slot and are masked later)
        const Expert& e = ws->lists[q][e_];
        const int l7 = lane & (MAX_TOK - 1);
        ntok = e.ntok;
        local = e.local;
        tokl = e.tok[l7];
        routel = e.route[l7];
        const float wtl = e.wt[l7];
        const float xs = __shfl_sync(0xffffffffu, xs13r, tokl & 31);
        // w13: the token's x13 scale; w2: the route weight (consumers apply
        // the x2 row's scale, which arrives with the row)
        fl = gr.kind == K_R0 ? xs : wtl;
      };
#ifdef TD_RECPF
      const auto prefetch = [&](int q_, int e_) {
        const Expert& e = ws->lists[q_][e_];
        const int l7 = lane & (MAX_TOK - 1);
        pf_e = e_;
        pf_q = q_;
        pf_ntok = e.ntok;
        pf_local = e.local;
        pf_tok = e.tok[l7];
        pf_route = e.route[l7];
        pf_wt = e.wt[l7];
      };
      if (routed) {
        if (pf_e != ei || pf_q != q) prefetch(q, ei);  // not prefetched
        ntok = pf_ntok;
        local = pf_local;
        tokl = pf_tok;
        routel = pf_route;
        const float xs = __shfl_sync(0xffffffffu, xs13r, tokl & 31);
        fl = gr.kind == K_R0 ? xs : pf_wt;
      }
      // the next group's record: its loads stay in flight across this
      // group's issue
      if (gq < (q ? len[1] : len[0])) {
        const Group gq_ = group_at(gq, (q ? n_tier[1] : n_tier[0]), q == 0 && sh, g1);
        if (gq_.kind == K_R0)
          prefetch(q, gq_.x / TILES0);
        else if (gq_.kind == K_R1)
          prefetch(q, gq_.x / (TILES1 / g1));
      }
#else
      if (routed) load_record(ei);
#endif
      for (int ci = 0; ci < gr.nch; ++ci, ++it) {
        const int s = it % STAGES, c = gr.c0 + ci;
        unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
        TD_V(const uint64_t td_e0 = td_now();)
        if (lane == 0 && it >= STAGES)
          mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
        TD_V(td_empty += td_now() - td_e0;)
#ifdef TD_UNIT_TRACE
        if (lane == 0) {
          s_issue[s] = td_now();
          td_record_at(td_ps++, 171, td_e0, s_issue[s], 0, gr.kind, ci);
        }
#else
#ifdef TD_UNIT_TRACE
        if (lane == 0) s_issue[s] = td_now();
#endif
#endif
        __syncwarp();
        const int hdr = gr.kind | q << 4 | (ci == 0) << 8 |
                        (ci == gr.nch - 1) << 9 | gr.nch << 16;
        if (gr.kind == K_R0) {
          if (lane < ntok) {
            sd_row[s][lane] = routel;
            sd_f[s][lane] = fl;
          }
          __syncwarp();
          if (lane == 0) {
            desc[s] = make_int4(hdr, ei, t0, ntok);
            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + min(ntok, XROWS) * XROW_BYTES0);
            tma_3d(dst, &tr.w[0], t0 * 2 * R0, c * KT0, local, &full[s]);
            tma_3d(dst + W_BYTES, &tr.s[0], t0 * R0, c * G0, local, &full[s]);
          }
          __syncwarp();
          if (lane < ntok && lane < XROWS)
            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
                     ws->x13 + static_cast<size_t>(tokl) * HIDDEN + c * CK0,
                     XROW_BYTES0, &full[s]);
        } else if (gr.kind == K_R1) {
#if TD_GUIDED > 0
          const int u = gr.x + ci, eu = u / TILES1, t = u - eu * TILES1;
          if (eu != ei) {
            ei = eu;
            load_record(ei);
          }
#else
          const int t = t0 + ci;
#endif
          if (lane < ntok) {
            sd_row[s][lane] = tokl;
            sd_f[s][lane] = fl;
          }
          __syncwarp();
          if (lane == 0) {
#if defined(TD_CTA_TRACE) && defined(TD_DEBUG_SUM)
            td_record(90 + q, ei, t | own << 16 | gi << 20, ntok, it);
#endif
            desc[s] = make_int4(hdr, ei, t, ntok);
            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + min(ntok, XROWS) * X2_COPY);
            tma_3d(dst, &tr.w[1], t * 256, 0, local, &full[s]);
            tma_3d(dst + W_BYTES, &tr.s[1], t * 128, 0, local, &full[s]);
          }
          // weights are in flight; only the activation rows wait for the
          // entry's ready (every copying lane acquires for itself)
          if (ei != last_ready) {
            TD_V(const uint64_t w0 = td_now();)
#ifndef TD_ABL_NOREADY  // timing ablation: w2 never waits (results invalid)
            while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
#endif
            TD_V(td_spin += td_now() - w0;)
#ifdef TD_UNIT_TRACE
            if (lane == 0) td_record_at(td_ps++, 172, w0, td_now(), 0, 0, 0);
#endif
            fence_proxy_async();
            last_ready = ei;
          }
          __syncwarp();
#ifdef TD_SPIN_EVERY_UNIT
          while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
          fence_proxy_async();
#endif
          if (lane < ntok && lane < XROWS)
            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
                     ws->x2 + static_cast<size_t>(routel) * X2_LD, X2_COPY,
                     &full[s]);
        } else {  // shared expert
          if (lane == 0) {
            desc[s] = make_int4(hdr, gr.x, c, 0);
            mbar_expect_tx(&full[s], W_BYTES + T * XS_BYTES);
            tma_2d(dst, &p.sw[gr.kind == K_S0 ? 0 : 1], c * CKS, gr.x * RS,
                   &full[s]);
          }
          if (gr.kind == K_S1 &&
              last_ready != -2) {  // every copying lane acquires
            TD_V(const uint64_t w0 = td_now();)
            while (ld_acquire(&ws->ready_s) != epoch) __nanosleep(32);
            TD_V(td_spin += td_now() - w0;)
            fence_proxy_async();
          }
          if (gr.kind == K_S1) last_ready = -2;
          __syncwarp();
          if (lane < T)
            bulk_g2s(dst + W_BYTES + lane * XS_BYTES,
                     (gr.kind == K_S0
                          ? ws->x13b + static_cast<size_t>(lane) * HIDDEN
                          : ws->x2s + static_cast<size_t>(lane) * INTER) +
                         c * CKS,
                     XS_BYTES, &full[s]);
        }
      }
#ifdef TD_NO_PREFETCH_CLAIM
      if (lane == 0) gn = atomicAdd(&ws->next[q], 1);
#endif
#ifdef TD_UNIT_TRACE
      const uint64_t td_cl = td_now();
#endif
#ifdef TD_RECPF
      gi = gq;
      gq = __shfl_sync(0xffffffffu, gn, 0);
#else
      gi = __shfl_sync(0xffffffffu, gn, 0);
#endif
      gk = kn;
#ifdef TD_UNIT_TRACE
      if (lane == 0) td_record_at(td_ps++, 173, td_cl, td_now(), 0, 0, 0);
#endif
    }
    if (lane == 0) {  // end of work
      const int s = it % STAGES;
      if (it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
      desc[s] = make_int4(K_END, 0, 0, 0);
      mbar_arrive(&full[s]);
    }
    TD_V(if (lane == 0)
             td_record(60 + own, td_p0, td_p0 + td_spin, n_tier[0], n_tier[1]);)
    TD_V(if (lane == 0) td_record(80 + own, td_p0, td_p0 + td_empty, it, 0);)
    return;
  }
```

## Consumer loop head (same file)
```cpp
  for (int it = 0;; ++it) {
    TD_V(const uint64_t td_w0 = td_now();)
    mbar_wait_a(full_u + 8 * s, ph);
#ifdef TD_UNIT_TRACE
    const uint64_t td_f = td_now();
#endif
    TD_V(td_wait += td_now() - td_w0;)
    const uint4 du = lds_v4(desc_u + 16 * s);
    const int4 d = make_int4(du.x, du.y, du.z, du.w);
    const int kind = d.x & 15;
    if (kind == K_END) break;
#ifdef TD_UNIT_TRACE
    if (warp == 0 && lane == 0)
      td_record_at(td_slot++, 120 + kind, s_issue[s], td_w0, td_f, d.w, (d.x >> 4) & 1);
#endif
    const int q = (d.x >> 4) & 1, last = (d.x >> 9) & 1, nch = d.x >> 16;
    const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
    const uint32_t st_u = ring_u + s * STAGE_BYTES;
    const uint32_t empty_s = empty_u + 8 * s;
    const Expert* experts = ws->lists[q];
    // this lane's two tokens of a routed unit: destination rows and scales
    const int ntok = d.w;
#ifdef TD_GLOOP
```

## Questions
1. Which option gives the most for the w2 phase at M=8 without hurting M=16/32? Rank A/B/C/D.
2. For A: is the smem-reservation + 2-END protocol correct (mbarrier phases with two producers each
   arriving expect_tx on positions they reserved; consumers' empty arrivals; the K_END case)?
   Pitfalls (e.g. warp scheduling: 3 warps on SMSP0 vs 2; producer starvation by consumer warps)?
3. For B: which of the listed edits actually shorten the critical path, in your estimate, and in which
   order? Is ld.relaxed + fence.acq_rel (once, after the flag is seen) a valid replacement for the
   ld.acquire spin before cp.async.bulk reads of x2 (fence.proxy.async is already issued after)?
4. Can I keep the compiler from putting the claim ATOMG on the same scoreboard as the record loads
   (asm volatile ordering, splitting the atomic into another warp, or a `red` + separate counter)?
Be concrete; code sketches welcome.
