# Astra review 11: v44-v46 of the sliced MoE layer_kernel (scheduler warp, ready probe, finisher warp)

Do NOT run any commands (your sandbox cannot run here). Review the code below for correctness first
(memory model, mbarrier phase logic, deadlock), then performance.

Context: same kernel as consult 10 (persistent sm_90a W4A16 MoE decode, 8 consumer warps + producer;
GRID 132; 4-stage TMA ring). Following your consult-10 advice (option C) I added:
- v44: scheduler warp (claims, stealing, decode, entry record) -> producer via a TD_SQ-slot smem FIFO.
  Bench: -1 to -6 us at every shape (M=8 38/4 91.1 -> 88.4; M=32 110/12 241.2 -> 234.9).
- v45: the scheduler probes the w2 entry's ready flag once (ld.acquire.gpu by scheduler lane 0) and
  publishes `ready` with the group; the producer skips its spin when set, but every copying lane still
  issues fence.proxy.async before its cp.async.bulk of x2 rows. Probe hit rate 95% at M=8. Neutral.
- v46: a finisher warp runs the w13 completion handoff that consumer warp 0 ran inline (it was ~4.8 us +
  1.5 us per CTA on warp 0 at M=8, ~19 us at M=32; warp 0 paces the ring since all 8 warps arrive on
  empty[s]). Consumers: warp 0 lane 0 waits hq_empty for the slot's previous use, writes (q, ei, nch);
  EVERY consumer thread (256) arrives on hq_full[k] after its own y13 red.add's (flush_rows ends with
  __syncwarp). The finisher lane 0 waits hq_full, reads info, frees the slot, then __threadfence +
  atomicAdd(done13); the winner __threadfence, __syncwarp, activates (all lanes read y13 via __ldcg),
  fence.proxy.async, __threadfence, __syncwarp, st.release.gpu(ready).
  My claim: no warp can be a whole R0 group (12 chunks, i.e. 12 ring stages) ahead of another through a
  4-stage ring (all 8 warps arrive on each empty[s]), so arrivals on hq_full[k] for slot generation g+HQ
  can never land in generation g's phase. Results pending.

Questions:
1. Is each handoff correct under the PTX memory model?
   (a) scheduler->producer FIFO: plain st.shared of the entry by scheduler lanes, __syncwarp,
       lane 0 mbarrier.arrive (default .release.cta); producer lane 0 try_wait.parity (default
       .acquire.cta), __syncwarp, all lanes ld.shared the entry.
   (b) v45: scheduler lane 0 ld.acquire.gpu(ready)==epoch -> the FIFO handoff (a) -> producer lanes
       fence.proxy.async -> cp.async.bulk reads of x2 rows written by another CTA before its
       st.release.gpu(ready). Is the acquire transitively valid for the producer's async-proxy reads?
   (c) v46: consumer threads' red.add.global (y13) -> per-thread mbarrier.arrive (release.cta) ->
       finisher lane 0 try_wait (acquire.cta) -> lane 0 fence (membar.gl / fence.sc.gpu via
       __threadfence) -> atomicAdd(done13). Remote winner: atomicAdd observes all counts, __threadfence,
       reads y13. Is cumulativity enough, or must every consumer thread fence itself?
2. The phase-generation argument for hq_full (and for sq_full / sq_empty).
3. Deadlock: the finisher only waits on hq_full; the producer waits on sq_full, empty[s] and (if not
   pre-ready) ready flags that some finisher (any CTA) publishes. Any cycle?
4. Performance: what next for M=8 (served case)? Current v45 M=8 38/4 trace per hot CTA: consume R0 34.7
   (31 units; math 0.61-0.64 us/unit), consume R1 19.0 (15 units; 0.9 math + 0.22 flush), waits R0 7.0,
   R1 4.6, head ~6, end ~4; kernel ~88 us vs HBM floor 61 us.

## mk_v44.py (generates td_v44.cu from td_v43.cu)
```python
"""td_v44.cu = td_v43.cu + TD_SCHED (default on; TD_NO_SCHED off): a scheduler
warp (warp 9, THREADS 320) owns the claims, work stealing, group decoding
(constant divisions only) and the entry record loads, and publishes whole
groups to the producer through a TD_SQ-slot shared-memory FIFO (mbarriers
sq_full / sq_empty; a slot is released once its group is issued, so at most
TD_SQ groups are claimed ahead). The producer only issues. Astra consult 10
(logs/codex-review10.md), option C. Under TD_SCHED the record prefetch of
v40 is off. Trace: 173 = producer's wait for the next group, 176 = scheduler
(claim issue, claim back, published), 177 = scheduler's wait for a slot."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v43.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#ifndef TD_NO_RECPF
  #define TD_RECPF
#endif''', '''#ifndef TD_NO_SCHED
  #define TD_SCHED
#endif
#ifndef TD_SQ
  #define TD_SQ 2  // scheduler -> producer FIFO depth (groups)
#endif
#if !defined(TD_NO_RECPF) && !defined(TD_SCHED)
  #define TD_RECPF
#endif''')
sub('''constexpr int THREADS = (CONSUMER_WARPS + 1) * 32;''', '''#ifdef TD_SCHED
constexpr int THREADS = (CONSUMER_WARPS + 2) * 32;  // + producer, scheduler
#else
constexpr int THREADS = (CONSUMER_WARPS + 1) * 32;
#endif''')
sub('''struct Group {
  int kind, x, c0,
      nch;  // x: R0 entry * TILES0 + tile, R1 entry * TILES1 + unit, S tile
};''', '''struct Group {
  int kind, x, c0,
      nch;  // x: R0 entry * TILES0 + tile, R1 entry * TILES1 + unit, S tile
};
// one claimed group, decoded, with its entry record (scheduler -> producer)
struct SchedEntry {
  int kind, x, c0, nch, q, ei, t0, ntok, local;
  int tok[MAX_TOK], route[MAX_TOK];
  float f[MAX_TOK];  // w13: the token's x13 scale; w2: the route weight
};''')
sub('''  __shared__ int s_last;
''', '''  __shared__ int s_last;
#ifdef TD_SCHED
  __shared__ SchedEntry sq[TD_SQ];
  __shared__ uint64_t sq_full[TD_SQ], sq_empty[TD_SQ];
#endif
''')
sub('''      mbar_init(&empty[s], CONSUMER_WARPS);
    }''', '''      mbar_init(&empty[s], CONSUMER_WARPS);
    }
#ifdef TD_SCHED
    for (int k = 0; k < TD_SQ; ++k) {
      mbar_init(&sq_full[k], 1);
      mbar_init(&sq_empty[k], 1);
    }
#endif''')

# --- the scheduler warp
sub('''  if (warp == CONSUMER_WARPS) {  // producer warp''', '''#ifdef TD_SCHED
  if (warp == CONSUMER_WARPS + 1) {  // scheduler warp
#ifdef TD_UNIT_TRACE
    unsigned td_ss = 0;  // lane 0: this CTA's scheduler records
    if (lane == 0) td_ss = atomicAdd(&td_trace_n, 512u);
#endif
    const float xs13r = lane < T ? ws->xs13[lane] : 0.f;
    int q = own, n = 0, k = 0;
    bool stolen = false;
    while (true) {
      if (n >= TD_SQ) {  // slot k is free once its last group is issued
        TD_V(const uint64_t td_w = td_now();)
        if (lane == 0) mbar_wait(&sq_empty[k], ((n / TD_SQ) - 1) & 1);
        __syncwarp();
#ifdef TD_UNIT_TRACE
        if (lane == 0) td_record_at(td_ss++, 177, td_w, td_now(), 0, 0, 0);
#endif
      }
      SchedEntry& e = sq[k];
      TD_V(const uint64_t td_c0 = td_now();)
      int gi = 0;
      if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
      gi = __shfl_sync(0xffffffffu, gi, 0);
      TD_V(const uint64_t td_c1 = td_now();)
      if (gi >= (q ? len[1] : len[0])) {
        bool steal = !stolen;
#ifdef TD_NO_STEAL
        steal = false;
#endif
#ifdef TD_NO_STEAL_COLD
        if (q == 0) steal = false;  // hot CTAs never take cold work
#endif
        if (steal) {
          stolen = true;
          q ^= 1;
          continue;
        }
        if (lane == 0) e.kind = K_END;
        __syncwarp();
        if (lane == 0) mbar_arrive(&sq_full[k]);
        break;
      }
      const Group gr =
          group_at(gi, q ? n_tier[1] : n_tier[0], q == 0 && sh, g1);
      int ei = 0, t0 = 0;
      if (gr.kind == K_R0) {
        ei = gr.x / TILES0;
        t0 = gr.x - ei * TILES0;
      } else if (gr.kind == K_R1) {
        if (g1 == 1) {
          ei = gr.x / TILES1;
          t0 = gr.x - ei * TILES1;
        } else {
          ei = gr.x / (TILES1 / GR1);
          t0 = (gr.x - ei * (TILES1 / GR1)) * GR1;
        }
      }
      if (gr.kind == K_R0 || gr.kind == K_R1) {
        // one round trip: every field at once (lanes past ntok read a valid
        // slot and are masked later)
        const Expert& x = ws->lists[q][ei];
        const int l7 = lane & (MAX_TOK - 1);
        const int ntok = x.ntok, local = x.local, tokl = x.tok[l7],
                  routel = x.route[l7];
        const float wtl = x.wt[l7];
        const float xs = __shfl_sync(0xffffffffu, xs13r, tokl & 31);
        if (lane < MAX_TOK) {
          e.tok[lane] = tokl;
          e.route[lane] = routel;
          e.f[lane] = gr.kind == K_R0 ? xs : wtl;
        }
        if (lane == 0) {
          e.ntok = ntok;
          e.local = local;
        }
      }
      if (lane == 0) {
        e.kind = gr.kind;
        e.x = gr.x;
        e.c0 = gr.c0;
        e.nch = gr.nch;
        e.q = q;
        e.ei = ei;
        e.t0 = t0;
      }
      __syncwarp();
#ifdef TD_UNIT_TRACE
      if (lane == 0) td_record_at(td_ss++, 176, td_c0, td_c1, td_now(), gr.kind, 0);
#endif
      if (lane == 0) mbar_arrive(&sq_full[k]);
      if (++k == TD_SQ) k = 0;
      ++n;
    }
    return;
  }
#endif
  if (warp == CONSUMER_WARPS) {  // producer warp''')

# --- producer head: take the next group from the FIFO
a = src.index('''    int q = own, last_ready = -1, it = 0;''')
b_mark = '''      for (int ci = 0; ci < gr.nch; ++ci, ++it) {
        const int s = it % STAGES, c = gr.c0 + ci;'''
b = src.index(b_mark)
old_head = src[a:b]
new_head = '''#ifdef TD_SCHED
    int last_ready = -1, it = 0, k = 0;
    uint32_t kph = 0;
    const int gi = 0;
    (void)gi;
    while (true) {
      TD_V(const uint64_t td_qw = td_now();)
      if (lane == 0) mbar_wait(&sq_full[k], kph);
      __syncwarp();
      const SchedEntry& e = sq[k];
      const Group gr = {e.kind, e.x, e.c0, e.nch};
      if (gr.kind == K_END) break;
#ifdef TD_UNIT_TRACE
      if (lane == 0) {
        td_record_at(td_ps++, 173, td_qw, td_now(), 0, 0, 0);
        td_record_at(td_ps++, 170, td_now(), 0, 0, 0, 0);
      }
#endif
      const int q = e.q, ei = e.ei, t0 = e.t0;
      int ntok = 0, local = 0, tokl = 0, routel = 0;
      float fl = 0.f;
      if (gr.kind == K_R0 || gr.kind == K_R1) {
        const int l7 = lane & (MAX_TOK - 1);
        ntok = e.ntok;
        local = e.local;
        tokl = e.tok[l7];
        routel = e.route[l7];
        fl = e.f[l7];
      }
      const Tier& tr = p.tier[q];
#else
''' + old_head + '''#endif
'''
src = src[:a] + new_head + src[b:]
sub('''          if (ei != last_ready) {''', '''          if ((ei | q << 16) != last_ready) {''')
sub('''            last_ready = ei;''', '''            last_ready = ei | q << 16;''')

# --- producer tail: release the slot
a = src.index('''#ifdef TD_NO_PREFETCH_CLAIM
      if (lane == 0) gn = atomicAdd(&ws->next[q], 1);
#endif''')
end_mark = '''      if (lane == 0) td_record_at(td_ps++, 173, td_cl, td_now(), 0, 0, 0);
#endif
    }'''
b = src.index(end_mark) + len(end_mark) - len("    }")
old_tail = src[a:b]
src = src[:a] + '''#ifdef TD_SCHED
      __syncwarp();
      if (lane == 0) mbar_arrive(&sq_empty[k]);
      if (++k == TD_SQ) {
        k = 0;
        kph ^= 1;
      }
#else
''' + old_tail + '''#endif
''' + src[b:]
(here / "td_v44.cu").write_text(src)
print("ok")
```

## mk_v45.py
```python
"""td_v45.cu = td_v44.cu + TD_SREADY (default on; TD_NO_SREADY off): for a w2
group the scheduler probes the entry's ready flag once (ld.acquire.gpu, lane
0) and publishes the result with the group. When it was ready the producer
skips its spin: the scheduler's acquire reaches the producer through the
FIFO's mbarrier (release / acquire at CTA scope) and __syncwarp; every copying
lane still issues fence.proxy.async before its bulk copies. When it was not
ready the producer spins as before (after the weight TMAs). Astra consult 10."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v44.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#ifndef TD_SQ''', '''#if defined(TD_SCHED) && !defined(TD_NO_SREADY)
  #define TD_SREADY
#endif
#ifndef TD_SQ''')
sub('''  int kind, x, c0, nch, q, ei, t0, ntok, local;''',
    '''  int kind, x, c0, nch, q, ei, t0, ntok, local, ready;''')
sub('''        e.q = q;
        e.ei = ei;
        e.t0 = t0;
      }''', '''        e.q = q;
        e.ei = ei;
        e.t0 = t0;
#ifdef TD_SREADY
        // one probe: if the entry is ready, this acquire is handed to the
        // producer by the FIFO barrier and its spin is skipped
        e.ready = gr.kind == K_R1 && ld_acquire(&ws->ready[q][ei]) == epoch;
#else
        e.ready = 0;
#endif
      }''')
sub('''      const int q = e.q, ei = e.ei, t0 = e.t0;''',
    '''      const int q = e.q, ei = e.ei, t0 = e.t0, pre_ready = e.ready;''')
sub('''          if ((ei | q << 16) != last_ready) {''', '''          if ((ei | q << 16) != last_ready) {
#ifdef TD_SCHED
           if (!pre_ready) {
#endif''')
sub('''            TD_V(td_spin += td_now() - w0;)
#ifdef TD_UNIT_TRACE
            if (lane == 0) td_record_at(td_ps++, 172, w0, td_now(), 0, 0, 0);
#endif''', '''            TD_V(td_spin += td_now() - w0;)
#ifdef TD_UNIT_TRACE
            if (lane == 0) td_record_at(td_ps++, 172, w0, td_now(), 0, 0, 0);
#endif
#ifdef TD_SCHED
           }
#endif''')
(here / "td_v45.cu").write_text(src)
print("ok")
src = (here / "td_v45.cu").read_text()
sub('''td_record_at(td_ss++, 176, td_c0, td_c1, td_now(), gr.kind, 0);''',
    '''td_record_at(td_ss++, 176, td_c0, td_c1, td_now(), gr.kind, e.ready);''')
(here / "td_v45.cu").write_text(src)
```

## mk_v46.py
```python
"""td_v46.cu = td_v45.cu + TD_FIN (default on with TD_SCHED; TD_NO_FIN off):
a finisher warp (THREADS 352) runs the w13 completion handoff that consumer
warp 0 ran inline (fence + done13 count; for the CTA that completes an entry:
activation + fence + ready release). Consumers hand off through a TD_HQ-slot
queue: warp 0 lane 0 writes (q, ei, nch) (after the slot's last use is
consumed: hq_empty), then EVERY consumer thread arrives on hq_full (count
256, each thread releases its own y13 reductions at CTA scope); the finisher
waits, takes the info, frees the slot, and does what warp 0 did (its fence
is cumulative over what the barrier made visible to it). A warp cannot get a
whole R0 group (12 chunks) ahead of another through the 4-stage ring, so
hq_full phases never mix generations. Consumers' K_END sends an end entry."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v45.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#ifndef TD_SQ''', '''#if defined(TD_SCHED) && !defined(TD_NO_FIN)
  #define TD_FIN
#endif
#ifndef TD_HQ
  #define TD_HQ 4  // consumer -> finisher handoff slots
#endif
#ifndef TD_SQ''')
sub('''#ifdef TD_SCHED
constexpr int THREADS = (CONSUMER_WARPS + 2) * 32;  // + producer, scheduler
#else''', '''#if defined(TD_FIN)
constexpr int THREADS = (CONSUMER_WARPS + 3) * 32;  // + producer, scheduler,
                                                    // finisher
#elif defined(TD_SCHED)
constexpr int THREADS = (CONSUMER_WARPS + 2) * 32;  // + producer, scheduler
#else''')
sub('''  __shared__ uint64_t sq_full[TD_SQ], sq_empty[TD_SQ];
#endif''', '''  __shared__ uint64_t sq_full[TD_SQ], sq_empty[TD_SQ];
#endif
#ifdef TD_FIN
  __shared__ int4 hq_info[TD_HQ];  // (q, entry, chunks, end)
  __shared__ uint64_t hq_full[TD_HQ], hq_empty[TD_HQ];
#endif''')
sub('''    for (int k = 0; k < TD_SQ; ++k) {
      mbar_init(&sq_full[k], 1);
      mbar_init(&sq_empty[k], 1);
    }
#endif''', '''    for (int k = 0; k < TD_SQ; ++k) {
      mbar_init(&sq_full[k], 1);
      mbar_init(&sq_empty[k], 1);
    }
#endif
#ifdef TD_FIN
    for (int k = 0; k < TD_HQ; ++k) {
      mbar_init(&hq_full[k], CONSUMER_WARPS * 32);
      mbar_init(&hq_empty[k], 1);
    }
#endif''')

# the finisher warp
sub('''#ifdef TD_SCHED
  if (warp == CONSUMER_WARPS + 1) {  // scheduler warp''', '''#ifdef TD_FIN
  if (warp == CONSUMER_WARPS + 2) {  // finisher warp
#ifdef TD_UNIT_TRACE
    unsigned td_fs = 0;  // lane 0: this CTA's finisher records
    if (lane == 0) td_fs = atomicAdd(&td_trace_n, 512u);
#endif
    int k = 0;
    uint32_t kph = 0;
    while (true) {
      TD_V(const uint64_t td_w = td_now();)
      if (lane == 0) mbar_wait(&hq_full[k], kph);
      __syncwarp();
      const int4 h = hq_info[k];
      __syncwarp();
      if (lane == 0) mbar_arrive(&hq_empty[k]);
      if (++k == TD_HQ) {
        k = 0;
        kph ^= 1;
      }
      if (h.w) break;
      const int q = h.x, ei = h.y, nch = h.z;
      TD_V(const uint64_t td_hc = td_now();)
      int done = 0;
      if (lane == 0) {
        __threadfence();
        done = atomicAdd(&ws->done13[q][ei], nch) + nch == UNITS0;
      }
      done = __shfl_sync(0xffffffffu, done, 0);
      TD_V(const uint64_t td_hd = td_now();)
      if (done) {
        __threadfence();
        __syncwarp();  // lane 0's acquire (the count) ordered before every
                       // lane's y13 reads
        const Expert& e = ws->lists[q][ei];
        const int n = e.ntok;
        for (int r = 0; r < n; ++r) activate_route(ws, e.route[r], lane);
        fence_proxy_async();
        __threadfence();
        __syncwarp();
        if (lane == 0) st_release(&ws->ready[q][ei], epoch);
      }
#ifdef TD_UNIT_TRACE
      if (lane == 0) {
        td_record_at(td_fs++, 178, td_w, td_hc, 0, 0, 0);
        td_record_at(td_fs++, 141 + done, td_hc, td_hd, td_now(), 0, 0);
      }
#endif
    }
    return;
  }
#endif
#ifdef TD_SCHED
  if (warp == CONSUMER_WARPS + 1) {  // scheduler warp''')

# consumers: handoff state, end entry
sub('''  int acq = -1;  // (tier, entry) whose ready this warp has acquired''',
    '''  int acq = -1;  // (tier, entry) whose ready this warp has acquired
#ifdef TD_FIN
  int hk = 0, hn = 0;  // handoff slot and count (same in every consumer)
  const auto hand_off = [&](int4 h) {
    if (warp == 0 && lane == 0) {
      if (hn >= TD_HQ) mbar_wait(&hq_empty[hk], ((hn / TD_HQ) - 1) & 1);
      hq_info[hk] = h;
    }
    mbar_arrive(&hq_full[hk]);  // every thread: its own reds, released
    if (++hk == TD_HQ) hk = 0;
    ++hn;
  };
#endif''')
sub('''    if (kind == K_END) break;''', '''    if (kind == K_END) {
#ifdef TD_FIN
      hand_off(make_int4(0, 0, 0, 1));
#endif
      break;
    }''')
sub('''        // warps 1..7 hand their reds to warp 0 and go on; warp 0 counts the
        // chunks and, if this CTA completed the entry, activates its routes
        // and publishes ready. Nobody else waits on the count's round trip.
        if (warp != 0) {''', '''#ifdef TD_FIN
        hand_off(make_int4(q, ei, nch, 0));
#ifdef TD_UNIT_TRACE
        if (warp == 0 && lane == 0)
          td_record_at(td_slot++, 140, td_ha, td_hb, td_now(), 0, 0);
#endif
        if (true) {  // the finisher counts; nothing more here
#else
        // warps 1..7 hand their reds to warp 0 and go on; warp 0 counts the
        // chunks and, if this CTA completed the entry, activates its routes
        // and publishes ready. Nobody else waits on the count's round trip.
        if (warp != 0) {
#endif''')
sub('''        if (warp != 0) {
#endif
          handoff_arrive();''', '''        if (warp != 0) {
#endif
#ifndef TD_FIN
          handoff_arrive();
#endif''')
(here / "td_v46.cu").write_text(src)
print("ok")
```

## activate_route and flush_rows (td_v46.cu)
```cpp
__device__ __forceinline__ void activate_route(Workspace* ws, int r, int lane) {
  float* __restrict__ yr = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
  float a[INTER / 32], m = 0.f;
#pragma unroll
  for (int q = 0; q < INTER / 32; ++q) {
    const float g = __ldcg(yr + q * 32 + lane),
                u = __ldcg(yr + INTER + q * 32 + lane);
    a[q] = __fdividef(g, 1.f + __expf(-g)) * u;
    m = fmaxf(m, fabsf(a[q]));
  }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  float scale;
  const float inv = row_scale(m, &scale);
  if (lane == 0) ws->xs2[r] = scale;
  __half* __restrict__ out = ws->x2 + static_cast<size_t>(r) * X2_LD;
  if (lane == 0) *reinterpret_cast<float*>(out + INTER) = scale;
#pragma unroll
  for (int q = 0; q < INTER / 32; ++q) {
    out[frag_slot(q * 32 + lane)] = __float2half_rn(a[q] * inv);
    yr[q * 32 + lane] = 0.f;
    yr[INTER + q * 32 + lane] = 0.f;
  }
}

// The shared expert's silu * up for all T tokens (bf16 fragments of x2s);
// y13s is zeroed for the next call.
__device__ __forceinline__ void activate_shared(Workspace* ws, int T, int warp,
                                                int lane) {
  for (int tok = warp; tok < T; tok += CONSUMER_WARPS) {
    float* __restrict__ yr = ws->y13s + static_cast<size_t>(tok) * 2 * INTER;
    __nv_bfloat16* __restrict__ out =
        ws->x2s + static_cast<size_t>(tok) * INTER;
#pragma unroll
    for (int q = 0; q < INTER / 32; ++q) {
      const int k = q * 32 + lane;
      const float g = __ldcg(yr + k), u = __ldcg(yr + INTER + k);
      out[frag_slot(k)] =
          __float2bfloat16_rn(__fdividef(g, 1.f + __expf(-g)) * u);
      yr[k] = 0.f;
      yr[INTER + k] = 0.f;
    }
__device__ __forceinline__ void flush_rows(float (*acc)[4], const float* fs,
                                           uint32_t scr, float* base,
                                           const int* rows4, int stride,
                                           int ntok, int g, int tq, int lane) {
#ifdef TD_ABL_NOFLUSH  // timing ablation: no stores / reds (acc kept live)
  float z = 0.f;
#pragma unroll
  for (int mb = 0; mb < 4; ++mb)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      z += acc[mb][i];
      acc[mb][i] = 0.f;
    }
  if (z == 1234.5f) red_add_if(base, z, true);
  return;
#endif
#pragma unroll
  for (int mb = 0; mb < 4; ++mb)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sts_f32(
          scr + ((2 * tq + (i & 1)) * SCR_LD + mb * 16 + g + (i >> 1) * 8) * 4,
          acc[mb][i] * fs[i & 1]);
      acc[mb][i] = 0.f;
    }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < MAX_TOK / 2; ++j) {  // token c = lane / 16 + 2 j
    const int k = lane + 32 * j;
    if (k < ntok * 16) {
      const int c = k >> 4, r4 = k & 15;
```
