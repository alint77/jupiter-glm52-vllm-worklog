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
