"""td_v43.cu = td_v42.cu + finer producer stamps under TD_UNIT_TRACE (no
effect untraced): 174 = (after the claim issue, after the group's index math,
after the entry record is in registers: the stamp takes the record as asm
inputs so it cannot run before the loads land); 175 = after the next
record's prefetch is issued (TD_RECPF)."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v42.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''__device__ __forceinline__ uint64_t td_now() {''', '''__device__ __forceinline__ uint64_t td_now_after(int a, int b, int c,
                                                 float d) {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;"
               : "=l"(t)
               : "r"(a), "r"(b), "r"(c), "f"(d));
  return t;
}
__device__ __forceinline__ uint64_t td_now() {''')
sub('''      Group gr = group_at(gi, (q ? n_tier[1] : n_tier[0]), q == 0 && sh, g1);''',
    '''      TD_V(const uint64_t td_sa = td_now();)
      Group gr = group_at(gi, (q ? n_tier[1] : n_tier[0]), q == 0 && sh, g1);''')
sub('''      const auto load_record = [&](int e_) {''', '''      TD_V(const uint64_t td_sb = td_now_after(ei, t0, gr.nch, 0.f);)
      const auto load_record = [&](int e_) {''')
sub('''#else
      if (routed) load_record(ei);
#endif
      for (int ci = 0; ci < gr.nch; ++ci, ++it) {''', '''#ifdef TD_UNIT_TRACE
      if (lane == 0) td_record_at(td_ps++, 175, td_now(), 0, 0, 0, 0);
#endif
#else
      if (routed) load_record(ei);
#endif
#ifdef TD_UNIT_TRACE
      {
        const uint64_t td_sc = td_now_after(tokl, local, ntok, fl);
        if (lane == 0) td_record_at(td_ps++, 174, td_sa, td_sb, td_sc, 0, 0);
      }
#endif
      for (int ci = 0; ci < gr.nch; ++ci, ++it) {''')
(here / "td_v43.cu").write_text(src)
print("ok")
