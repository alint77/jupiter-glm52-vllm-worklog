"""td_v38.cu = td_v37.cu with TD_MMA_NV, TD_MB2 and TD_GLOOP on by default
(TD_V37_OFF turns them off) + TD_GUIDED=K: guided self-scheduling of the
routed w2 queue. Once a producer's claims are in the R1 region it claims
k = clamp(remaining / (2 GRID), 1, K) units at once; a claim may span two
entries (the record is reloaded when the entry changes)."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v37.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#ifndef TD_COLD_CTAS''', '''#ifndef TD_V37_OFF  // consumer changes of v37 (Astra consult 9)
  #define TD_MMA_NV
  #define TD_MB2
  #define TD_GLOOP
#endif
#ifndef TD_GUIDED
  #define TD_GUIDED 0  // max routed w2 units per claim, guided (0: GR1)
#endif
#ifndef TD_COLD_CTAS''')
sub('''static_assert(TILES1 % GR1 == 0, "w2 groups tile an entry");''',
    '''static_assert(TILES1 % GR1 == 0, "w2 groups tile an entry");
static_assert(TD_GUIDED == 0 || GR1 == 1, "guided claims index single units");''')

# claim sizes travel with the claim
sub('''    int gi = 0;
    if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
    gi = __shfl_sync(0xffffffffu, gi, 0);
    while (true) {''', '''    int gi = 0, gk = 1;  // the current claim and its unit count
    if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
    gi = __shfl_sync(0xffffffffu, gi, 0);
    while (true) {''')
sub('''        q ^= 1;
        last_ready = -1;
        if (lane == 0) gi = atomicAdd(&ws->next[q], 1);''', '''        q ^= 1;
        last_ready = -1;
        gk = 1;
        if (lane == 0) gi = atomicAdd(&ws->next[q], 1);''')
sub('''      int gn = 0;
#ifndef TD_NO_PREFETCH_CLAIM
      if (lane == 0) gn = atomicAdd(&ws->next[q], 1);
#endif
      const Group gr = group_at(gi, n_tier[q], q == 0 && sh);''', '''      int gn = 0, kn = 1;
#if TD_GUIDED > 0
      {
        // past the start of R1 the counter only moves through R1 units
        const int r1s = len[q] - n_tier[q] * TILES1;
        if (gi >= r1s)
          kn = max(1, min(TD_GUIDED, (len[q] - gi - gk) / (2 * GRID)));
      }
#endif
#ifndef TD_NO_PREFETCH_CLAIM
      if (lane == 0) gn = atomicAdd(&ws->next[q], kn);
#endif
      Group gr = group_at(gi, n_tier[q], q == 0 && sh);
#if TD_GUIDED > 0
      if (gr.kind == K_R1) gr.nch = min(gk, len[q] - gi);
#endif''')

# the record load as a lambda, reused when a guided claim crosses entries
sub('''      if (routed) {
        // one round trip: every field at once (lanes past ntok read a valid
        // slot and are masked later)
        const Expert& e = ws->lists[q][ei];
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
      }''', '''      const auto load_record = [&](int e_) {
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
      if (routed) load_record(ei);''')
sub('''        } else if (gr.kind == K_R1) {
          const int t = t0 + ci;''', '''        } else if (gr.kind == K_R1) {
#if TD_GUIDED > 0
          const int u = gr.x + ci, eu = u / TILES1, t = u - eu * TILES1;
          if (eu != ei) {
            ei = eu;
            load_record(ei);
          }
#else
          const int t = t0 + ci;
#endif''')
sub('''      gi = __shfl_sync(0xffffffffu, gn, 0);''', '''      gi = __shfl_sync(0xffffffffu, gn, 0);
      gk = kn;''')
(here / "td_v38.cu").write_text(src)
print("ok")
