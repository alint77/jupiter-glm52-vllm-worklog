"""td_v42.cu = td_v41.cu + TD_ABL_STATIC (timing ablation, but a valid
schedule): no claim atomics in the loop; each CTA takes its queue's groups
with a static stride (implies TD_NO_STEAL). Measures what the claim atomic
costs the producer: nvcc puts it on the same scoreboard as the entry record
loads, so every group waits for the contended atomic before its first TMA."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v41.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#ifndef TD_NO_RECPF''', '''#ifdef TD_ABL_STATIC
  #define TD_NO_STEAL
#endif
#ifndef TD_NO_RECPF''')
sub('''#endif
    gi = __shfl_sync(0xffffffffu, gi, 0);
    while (true) {''', '''#endif
    gi = __shfl_sync(0xffffffffu, gi, 0);
#ifdef TD_ABL_STATIC
    const int ns = own ? cold_ctas : GRID - cold_ctas;
    gi = own ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.x) - cold_ctas;
  #ifdef TD_RECPF
    gq = gi + ns;
  #endif
#endif
    while (true) {''')
sub('''      if (lane == 0) gn = atomicAdd(&ws->next[q], kn);''', '''#ifdef TD_ABL_STATIC
  #ifdef TD_RECPF
      gn = gq + ns;
  #else
      gn = gi + ns;
  #endif
#else
      if (lane == 0) gn = atomicAdd(&ws->next[q], kn);
#endif''')
(here / "td_v42.cu").write_text(src)
print("ok")
