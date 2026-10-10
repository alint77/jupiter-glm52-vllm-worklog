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
