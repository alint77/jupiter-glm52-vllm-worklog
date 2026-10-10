"""td_v47.cu = td_v46.cu +
TD_SPIPE (default on; TD_NO_SPIPE off): the scheduler issues the NEXT group's
claim atomic before it loads the current group's record and probes its
ready flag, and issues the ready probe right after the record loads, so the
three round trips per group overlap instead of running back to back (v46:
claim 0.29 + record / probe / publish 1.09 us per w2 group at M=8).
TD_FIN_AR: the finisher's three __threadfence (MEMBAR.SC.GPU) become
fence.acq_rel.gpu (the fence-atomic release / acquire patterns only need
acq_rel)."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v46.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#ifndef TD_SQ''', '''#if defined(TD_SCHED) && !defined(TD_NO_SPIPE)
  #define TD_SPIPE
#endif
#ifndef TD_SQ''')
sub('''    int q = own, n = 0, k = 0;
    bool stolen = false;
    while (true) {''', '''    int q = own, n = 0, k = 0;
    bool stolen = false;
#ifdef TD_SPIPE
    int gi = 0;  // this group's claim; the next one is issued before its loads
    if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
    gi = __shfl_sync(0xffffffffu, gi, 0);
#endif
    while (true) {''')
sub('''      TD_V(const uint64_t td_c0 = td_now();)
      int gi = 0;
      if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
      gi = __shfl_sync(0xffffffffu, gi, 0);
      TD_V(const uint64_t td_c1 = td_now();)''', '''      TD_V(const uint64_t td_c0 = td_now();)
#ifdef TD_SPIPE
      int gn = 0;
      if (lane == 0) gn = atomicAdd(&ws->next[q], 1);
#else
      int gi = 0;
      if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
      gi = __shfl_sync(0xffffffffu, gi, 0);
#endif
      TD_V(const uint64_t td_c1 = td_now();)''')
sub('''        if (steal) {
          stolen = true;
          q ^= 1;
          continue;
        }''', '''        if (steal) {
          stolen = true;
          q ^= 1;
#ifdef TD_SPIPE
          if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
          gi = __shfl_sync(0xffffffffu, gi, 0);
#endif
          continue;
        }''')
sub('''      if (gr.kind == K_R0 || gr.kind == K_R1) {
        // one round trip: every field at once (lanes past ntok read a valid
        // slot and are masked later)
        const Expert& x = ws->lists[q][ei];
        const int l7 = lane & (MAX_TOK - 1);
        const int ntok = x.ntok, local = x.local, tokl = x.tok[l7],
                  routel = x.route[l7];
        const float wtl = x.wt[l7];''', '''      int rdy = 0;
      if (gr.kind == K_R0 || gr.kind == K_R1) {
        // one round trip: every field at once (lanes past ntok read a valid
        // slot and are masked later)
        const Expert& x = ws->lists[q][ei];
        const int l7 = lane & (MAX_TOK - 1);
        const int ntok = x.ntok, local = x.local, tokl = x.tok[l7],
                  routel = x.route[l7];
        const float wtl = x.wt[l7];
#if defined(TD_SPIPE) && defined(TD_SREADY)
        // the ready probe in flight with the record loads
        if (lane == 0 && gr.kind == K_R1)
          rdy = ld_acquire(&ws->ready[q][ei]) == epoch;
#endif''')
sub('''        e.ready = gr.kind == K_R1 && ld_acquire(&ws->ready[q][ei]) == epoch;''',
    '''#ifdef TD_SPIPE
        e.ready = rdy;
#else
        e.ready = gr.kind == K_R1 && ld_acquire(&ws->ready[q][ei]) == epoch;
#endif''')
sub('''      if (++k == TD_SQ) k = 0;
      ++n;
    }''', '''      if (++k == TD_SQ) k = 0;
      ++n;
#ifdef TD_SPIPE
      gi = __shfl_sync(0xffffffffu, gn, 0);
#endif
    }''')
# finisher fences
a = src.index('''  if (warp == CONSUMER_WARPS + 2) {  // finisher warp''')
b = src.index('''#ifdef TD_SCHED
  if (warp == CONSUMER_WARPS + 1) {  // scheduler warp''')
fin = src[a:b]
assert fin.count("__threadfence();") == 3
fin = fin.replace("__threadfence();", "TD_FIN_FENCE();")
src = src[:a] + fin + src[b:]
sub('''__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}''', '''__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
#ifdef TD_FIN_AR
  #define TD_FIN_FENCE fence_acq_rel_gpu
#else
  #define TD_FIN_FENCE __threadfence
#endif''')
(here / "td_v47.cu").write_text(src)
print("ok")
