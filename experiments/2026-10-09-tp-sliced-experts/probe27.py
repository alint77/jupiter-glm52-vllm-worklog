"""td_v27 TD_CTA_TRACE: the last call's route_prep span vs the layer CTAs' start
(us from route_prep's first entry)."""
import statistics as st
import sys

import torch

r = torch.load(sys.argv[1]).tolist()
recs = [(w0 & 0xFF, (w0 >> 8) & 0xFF, a, b, c) for w0, a, b, c in r]
rp = sorted((x for x in recs if x[0] == 50), key=lambda x: x[2])
t_last = rp[-1][2]
rp = [x for x in rp if x[2] > t_last - 20000]  # the last call's blocks
t0 = min(x[2] for x in rp)
us = lambda v: (v - t0) / 1e3  # noqa: E731
f = lambda xs: f"{st.median(xs):6.1f} / {max(xs):6.1f}"  # noqa: E731
print(f"route_prep {len(rp)} blocks: past PDL wait {f([us(x[3]) for x in rp])}, exit {f([us(x[4]) for x in rp])}")
cta = [x for x in recs if x[0] in (0, 1, 10, 11) and x[2] > t0]
print(f"layer CTAs {len(cta)}: past PDL wait {f([us(x[2]) for x in cta])}, end {f([us(x[4]) for x in cta])}")
pr = [x for x in recs if x[0] in (60, 61) and x[2] > t0]
print(f"producers: first issue {f([us(x[2]) for x in pr])}, last issue {f([us(x[4]) for x in pr])}")
fi = [x for x in recs if x[0] == 54 and x[2] > t0]
if fi:
    print(f"finalize {len(fi)} blocks: entry {f([us(x[2]) for x in fi])}, past wait {f([us(x[3]) for x in fi])}, exit {f([us(x[4]) for x in fi])}")
