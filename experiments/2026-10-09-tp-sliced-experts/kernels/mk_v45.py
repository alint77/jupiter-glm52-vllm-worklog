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
