"""td_v48.cu = td_v47.cu with Astra review 11's fixes (logs/codex-review11.md):
- hq_full reuse: every consumer warp's lane 0 waits hq_empty for the slot's
  previous use before arriving, so no arrival enters a phase whose
  predecessor has not been waited on (PTX mbarrier phase rule); with that
  gate the arrival is one per warp (count CONSUMER_WARPS) after __syncwarp.
- the producer reads e.kind and stops at K_END before reading the rest of a
  slot (the end entry only writes kind)."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v47.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''      mbar_init(&hq_full[k], CONSUMER_WARPS * 32);''',
    '''      mbar_init(&hq_full[k], CONSUMER_WARPS);''')
sub('''  const auto hand_off = [&](int4 h) {
    if (warp == 0 && lane == 0) {
      if (hn >= TD_HQ) mbar_wait(&hq_empty[hk], ((hn / TD_HQ) - 1) & 1);
      hq_info[hk] = h;
    }
    mbar_arrive(&hq_full[hk]);  // every thread: its own reds, released''',
    '''  const auto hand_off = [&](int4 h) {
    // every warp waits for the slot's previous use to be taken before it
    // arrives (no arrival may enter a phase whose predecessor nobody waited
    // on); __syncwarp gathers the warp's reds and warp 0's info for lane 0's
    // release
    if (lane == 0) {
      if (hn >= TD_HQ) mbar_wait(&hq_empty[hk], ((hn / TD_HQ) - 1) & 1);
      if (warp == 0) hq_info[hk] = h;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&hq_full[hk]);''')
sub('''      const SchedEntry& e = sq[k];
      const Group gr = {e.kind, e.x, e.c0, e.nch};
      if (gr.kind == K_END) break;''', '''      const SchedEntry& e = sq[k];
      if (e.kind == K_END) break;
      const Group gr = {e.kind, e.x, e.c0, e.nch};''')
(here / "td_v48.cu").write_text(src)
print("ok")
