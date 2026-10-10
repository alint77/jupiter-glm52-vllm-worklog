"""td_v53.cu = td_v52.cu with
- Astra review 12 adopted as defaults: TD_ONEREL, TD_ATOM_AR, no writer-side
  proxy fence in the finisher (readers fence after their acquire), y13 row
  pointers built only for valid routes.
- TD_NOAHEAD: the scheduler claims no group ahead of a multi-chunk one
  (S0/R0/S1): after publishing it, it waits until the producer has issued it,
  then claims. v52's entry timeline: w13 groups waited 15-25 us behind their
  CTA's current group while other CTAs ran short of w2 work.
- TD_RLIST: w2 work in ready order. The finisher appends each ready entry to
  a per-tier list (tail atomic, then st.release.b64 {epoch, entry}); a w2
  claim's entry slot j is the j-th entry to become ready, and the scheduler
  acquires the slot (the producer then skips its spin, as with SREADY). No w2
  claim waits behind a slow entry while a later one is ready (v52: 46 claims
  spun 178 CTA-us on one entry)."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v52.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#ifndef TD_NO_ZLATE''', '''#ifndef TD_V51_FENCES  // Astra review 12
  #define TD_ONEREL
  #define TD_ATOM_AR
#endif
#if defined(TD_NOAHEAD) && !defined(TD_SPIPE_REQ)
  #define TD_SPIPE_REQ
#endif
#ifndef TD_NO_ZLATE''')
sub('''#if defined(TD_SCHED) && !defined(TD_NO_SPIPE)''',
    '''#if defined(TD_RLIST) && !(defined(TD_SCHED) && defined(TD_FIN))
  #error "TD_RLIST needs the scheduler and finisher warps"
#endif
#if defined(TD_SCHED) && !defined(TD_NO_SPIPE)''')
# workspace: the ready lists
sub('''  int ready_s;  // epoch once x2s is written
  int T;
};''', '''  int ready_s;  // epoch once x2s is written
  int T;
  int rl_tail[2];  // per tier: entries appended to rl, zeroed by route_prep
  // per tier, in ready order: {epoch, entry} once the entry's x2 rows are
  // written
  alignas(8) unsigned long long rl[2][MAX_LIST];
};''')
sub('''    ws->next[0] = ws->next[1] = 0;''', '''    ws->next[0] = ws->next[1] = 0;
    ws->rl_tail[0] = ws->rl_tail[1] = 0;''')
sub('''__device__ __forceinline__ void fence_proxy_async() {''', '''__device__ __forceinline__ unsigned long long ld_acquire64(
    const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release64(unsigned long long* p,
                                             unsigned long long v) {
  asm volatile("st.release.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v)
               : "memory");
}
__device__ __forceinline__ void fence_proxy_async() {''')
# Astra 12: no out-of-range pointer arithmetic for inactive routes
sub('''      const int r = __shfl_sync(0xffffffffu, routel, (r0 + j) & 31);
      yr[j] = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
      m[j] = 0.f;
      if (r0 + j < n) {''', '''      const int r = __shfl_sync(0xffffffffu, routel, (r0 + j) & 31);
      m[j] = 0.f;
      if (r0 + j < n) {
        yr[j] = ws->y13 + static_cast<size_t>(r) * 2 * INTER;''')
# finisher: no writer proxy fence; ready list
sub('''        fence_proxy_async();
#ifndef TD_ONEREL
        TD_FIN_FENCE();
#endif
        __syncwarp();
#if defined(TD_UNIT_TRACE) && TD_ACTK > 1''', '''#ifdef TD_WPROXY
        fence_proxy_async();
#endif
#ifndef TD_ONEREL
        TD_FIN_FENCE();
#endif
        __syncwarp();
#if defined(TD_UNIT_TRACE) && TD_ACTK > 1''')
sub('''        if (lane == 0) st_release(&ws->ready[q][ei], epoch);
#if defined(TD_ZLATE) && TD_ACTK > 1''', '''#ifdef TD_RLIST
        if (lane == 0)
          st_release64(&ws->rl[q][atomicAdd(&ws->rl_tail[q], 1)],
                       static_cast<unsigned long long>(epoch) << 32 |
                           static_cast<unsigned>(ei));
#else
        if (lane == 0) st_release(&ws->ready[q][ei], epoch);
#endif
#if defined(TD_ZLATE) && TD_ACTK > 1''')
# scheduler: NOAHEAD
sub('''#ifdef TD_SPIPE
      int gn = 0;
      if (lane == 0) gn = atomicAdd(&ws->next[q], 1);
#else''', '''#ifdef TD_SPIPE
      int gn = 0;
#ifdef TD_NOAHEAD
      // a multi-chunk group is not claimed past: its successor is claimed
      // once the producer has issued it
      const bool heavy =
#ifdef TD_NOAHEAD_HOT  // cold groups (C2C) keep their lookahead
          q == 0 &&
#endif
          gi < (q ? len[1] : len[0]) &&
          group_at(gi, q ? n_tier[1] : n_tier[0], q == 0 && sh, g1).kind !=
              K_R1;
      if (lane == 0 && !heavy) gn = atomicAdd(&ws->next[q], 1);
#else
      if (lane == 0) gn = atomicAdd(&ws->next[q], 1);
#endif
#else''')
sub('''      if (lane == 0) mbar_arrive(&sq_full[k]);
      if (++k == TD_SQ) k = 0;
      ++n;''', '''      if (lane == 0) mbar_arrive(&sq_full[k]);
#ifdef TD_NOAHEAD
      if (heavy && lane == 0) {
        mbar_wait(&sq_empty[k], (n / TD_SQ) & 1);
        gn = atomicAdd(&ws->next[q], 1);
      }
#endif
      if (++k == TD_SQ) k = 0;
      ++n;''')
# scheduler: RLIST maps the w2 group's entry slot to the slot-th ready entry
sub('''      int rdy = 0;
      if (gr.kind == K_R0 || gr.kind == K_R1) {''', '''      int rdy = 0;
#ifdef TD_RLIST
      if (gr.kind == K_R1) {
        // entry slot ei is the ei-th entry of this tier to become ready
        unsigned long long v = 0;
        if (lane == 0) {
          while ((v = ld_acquire64(&ws->rl[q][ei])) >> 32 !=
                 static_cast<unsigned>(epoch))
            __nanosleep(32);
          rdy = 1;
        }
        ei = __shfl_sync(0xffffffffu, static_cast<int>(v & 0xffffffffu), 0);
      }
#endif
      if (gr.kind == K_R0 || gr.kind == K_R1) {''')
sub('''#if defined(TD_SPIPE) && defined(TD_SREADY)
        // the ready probe in flight with the record loads
        if (lane == 0 && gr.kind == K_R1)''', '''#if defined(TD_SPIPE) && defined(TD_SREADY) && !defined(TD_RLIST)
        // the ready probe in flight with the record loads
        if (lane == 0 && gr.kind == K_R1)''')
src = src.replace("#if defined(TD_NOAHEAD) && !defined(TD_SPIPE_REQ)\n  #define TD_SPIPE_REQ\n#endif\n", "")
(here / "td_v53.cu").write_text(src)
print("ok")
