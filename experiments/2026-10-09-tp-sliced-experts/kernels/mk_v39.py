"""td_v39.cu = td_v38.cu with the routed w2 claim size chosen per call:
1 unit for T <= TD_GR1_T tokens (default 8), else TD_GR1 (default 2)."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v38.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#ifndef TD_GR1
  #define TD_GR1 1  // routed w2 units per claim; 2-4 within noise
#endif
constexpr int GR1 =
    TD_GR1;  // routed w2 units per group (consecutive tiles, one entry)
static_assert(TILES1 % GR1 == 0, "w2 groups tile an entry");
static_assert(TD_GUIDED == 0 || GR1 == 1, "guided claims index single units");''',
    '''// routed w2 units per claim (consecutive tiles of one entry): GR1 above
// GR1_T tokens, else 1. Larger claims amortize the producer's claim, record
// and ready round trips (M=16/32 -5..7%); at M=8 they unbalance the tail.
#ifndef TD_GR1
  #define TD_GR1 2
#endif
#ifndef TD_GR1_T
  #define TD_GR1_T 8
#endif
constexpr int GR1 = TD_GR1;
static_assert(TILES1 % GR1 == 0, "w2 groups tile an entry");
static_assert(TD_GUIDED == 0, "v39 sizes w2 claims per call, not guided");''')
sub('''__device__ __forceinline__ int queue_len(int n, bool sh) {
  return (sh ? TILES_S0 * SG0 + TILES_S1 : 0) +
         n * (TILES0 * R0S + TILES1 / GR1);
}
__device__ __forceinline__ Group group_at(int gi, int n, bool sh) {''',
    '''__device__ __forceinline__ int queue_len(int n, bool sh, int g1) {
  return (sh ? TILES_S0 * SG0 + TILES_S1 : 0) +
         n * (TILES0 * R0S + TILES1 / g1);
}
__device__ __forceinline__ Group group_at(int gi, int n, bool sh, int g1) {''')
sub('''  return {K_R1, gi, 0, GR1};''', '''  return {K_R1, gi, 0, g1};''')
sub('''  const int len[2] = {queue_len(n_tier[0], sh), queue_len(n_tier[1], false)};''',
    '''  const int g1 = T > TD_GR1_T ? GR1 : 1;
  const int len[2] = {queue_len(n_tier[0], sh, g1),
                      queue_len(n_tier[1], false, g1)};''')
sub('''      Group gr = group_at(gi, n_tier[q], q == 0 && sh);''',
    '''      Group gr = group_at(gi, n_tier[q], q == 0 && sh, g1);''')
sub('''        ei = gr.x / (TILES1 / GR1);
        t0 = (gr.x - ei * (TILES1 / GR1)) * GR1;''', '''        ei = gr.x / (TILES1 / g1);
        t0 = (gr.x - ei * (TILES1 / g1)) * g1;''')
(here / "td_v39.cu").write_text(src)
print("ok")
