"""Phase times of td_v29 TD_PROBE calls (kdev once --trace): per call, us from
the first CTA's entry, median / max over CTAs."""
import statistics as st
import sys

import torch

r = torch.load(sys.argv[1]).tolist()
recs = [(w0 & 0xFF, (w0 >> 8) & 0xFFFF, w0 >> 24, a, b, c) for w0, a, b, c in r]
ent = sorted(x[3] for x in recs if x[0] == 1)
# calls: entry times cluster; split at gaps > 20 us
calls, cur = [], [ent[0]]
for t in ent[1:]:
    if t - cur[-1] > 20000:
        calls.append(cur)
        cur = []
    cur.append(t)
calls.append(cur)
for ci, c in enumerate(calls):
    t0, t1 = c[0], c[-1] + 400000
    inn = lambda x: t0 <= x[3] <= t1  # noqa: E731
    k1 = [x for x in recs if x[0] == 1 and inn(x)]
    k2 = [x for x in recs if x[0] == 2 and t0 <= x[3] <= t1]
    k3 = [x for x in recs if x[0] == 3 and t0 <= x[3] <= t1]
    k4 = [x for x in recs if x[0] == 4 and t0 <= x[3] <= t1 + 400000]
    us = lambda v: (v - t0) / 1e3  # noqa: E731
    f = lambda xs: f"{st.median(xs):6.1f} / {max(xs):6.1f}" if xs else "-"  # noqa: E731
    pp = [us(x[5]) for x in k1 if x[5]]
    print(f"call {ci}: {len(k1)} CTAs | entry spread {us(max(x[3] for x in k1)):5.1f} | past PDL wait {f([us(x[4]) for x in k1])}"
          f" | prep published {f(pp)} (n={len(pp)}) | prep seen {f([us(x[3]) for x in k2])}"
          f" | first issue {f([us(x[4]) for x in k2 if x[4]])} | producer end {f([us(x[5]) for x in k2])}"
          f" | consumer first full {f([us(x[3]) for x in k3])} end {f([us(x[4]) for x in k3])}"
          + (f" | finalize entry {us(k4[0][3]):.1f} wait {us(k4[0][4]):.1f} end {us(k4[0][5]):.1f}" if k4 else ""))
