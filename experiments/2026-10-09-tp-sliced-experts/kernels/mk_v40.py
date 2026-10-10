"""td_v40.cu = td_v39.cu + TD_RECPF: the producer claims two groups ahead and
prefetches the next group's entry record while it issues the current one (the
record round trip was ~0.9 us of a ~2.4 us per-unit producer path at M=8, GR1=1).
The per-unit 90+q CTA trace record (a global atomic on the producer's path)
now needs TD_DEBUG_SUM."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v39.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#ifndef TD_GUIDED''', '''#ifndef TD_NO_RECPF
  #define TD_RECPF
#endif
#ifndef TD_GUIDED''')

# claims: the current group and the next one
sub('''    int gi = 0, gk = 1;  // the current claim and its unit count
    if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
    gi = __shfl_sync(0xffffffffu, gi, 0);''', '''    int gi = 0, gk = 1;  // the current claim and its unit count
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
    gi = __shfl_sync(0xffffffffu, gi, 0);''')
sub('''        gk = 1;
        if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
        gi = __shfl_sync(0xffffffffu, gi, 0);''', '''        gk = 1;
#ifdef TD_RECPF
        if (lane == 0) {
          gi = atomicAdd(&ws->next[q], 1);
          gq = atomicAdd(&ws->next[q], 1);
        }
        gq = __shfl_sync(0xffffffffu, gq, 0);
#else
        if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
#endif
        gi = __shfl_sync(0xffffffffu, gi, 0);''')

sub('''      if (routed) load_record(ei);''', '''#ifdef TD_RECPF
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
      if (gq < len[q]) {
        const Group gq_ = group_at(gq, n_tier[q], q == 0 && sh, g1);
        if (gq_.kind == K_R0)
          prefetch(q, gq_.x / TILES0);
        else if (gq_.kind == K_R1)
          prefetch(q, gq_.x / (TILES1 / g1));
      }
#else
      if (routed) load_record(ei);
#endif''')
sub('''#ifdef TD_CTA_TRACE
            td_record(90 + q, ei, t | own << 16 | gi << 20, ntok, it);
#endif''', '''#if defined(TD_CTA_TRACE) && defined(TD_DEBUG_SUM)
            td_record(90 + q, ei, t | own << 16 | gi << 20, ntok, it);
#endif''')
sub('''      gi = __shfl_sync(0xffffffffu, gn, 0);
      gk = kn;''', '''#ifdef TD_RECPF
      gi = gq;
      gq = __shfl_sync(0xffffffffu, gn, 0);
#else
      gi = __shfl_sync(0xffffffffu, gn, 0);
#endif
      gk = kn;''')
(here / "td_v40.cu").write_text(src)
print("ok")
