"""Timeline of one traced call of the persistent layer kernel (v1+): per
group (hot / cold CTAs) the spread of ready (past PDL wait), w13-done and exit
times, and the producer's total wait on ready flags; route_prep / finalize
spans. Times in us from route_prep's first entry.

    tl2.py t.pt
"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph = r[:, 0] & 0xFF
nh = ((r[:, 0] >> 32) & 0xFFFF).max()
nc = ((r[:, 0] >> 48) & 0xFFFF).max()
t0 = r[ph == 50, 1].min() if (ph == 50).any() else r[:, 1].min()
us = lambda x: (x - t0) / 1e3  # noqa: E731
q = lambda x: f"{x.min():6.1f} p10 {np.percentile(x, 10):6.1f} p50 {np.median(x):6.1f} p90 {np.percentile(x, 90):6.1f} max {x.max():6.1f}"  # noqa: E731
print(f"hot entries {nh}, cold entries {nc}")
for p, name in ((50, "route_prep"), (54, "finalize")):
    m = ph == p
    if m.any():
        print(f"{name:10s} entry {us(r[m, 1].min()):6.1f}  past wait {us(r[m, 2].max()):6.1f}  exit {us(r[m, 3].max()):6.1f}")
# CTA record: {kind, ready (past PDL wait), last routed w13 flush, exit}
for p, name in ((0, "hot"), (1, "cold")):
    m = ph == p
    if not m.any():
        continue
    print(f"{name} CTAs {m.sum()}:")
    print(f"   ready      {q(us(r[m, 1]))}")
    w = r[m, 2][r[m, 2] > 0]
    if len(w):
        print(f"   w13 done   {q(us(w))}")
    print(f"   exit       {q(us(r[m, 3]))}")
    prod = ph == 60 + p
    if prod.any():
        spin = (r[prod, 2] - r[prod, 1]) / 1e3
        print(f"   producer ready-wait us {q(spin)}")
for p, name in ((0, "hot"), (1, "cold")):
    c = ph == 70 + p
    e = ph == 80 + p
    if c.any():
        span = (r[ph == p, 3] - r[ph == p, 1]) / 1e3
        wait = (r[c, 2] - r[c, 1]) / 1e3
        print(f"{name}: consumer warp 0 waiting on full stages {np.median(wait):.1f} of {np.median(span):.1f} us (median)")
    if e.any():
        ew = (r[e, 2] - r[e, 1]) / 1e3
        units = (r[e, 0] >> 32) & 0xFFFF
        print(f"{name}: producer blocked on empty stages {np.median(ew):.1f} us, units p50 {np.median(units):.0f}")
for p, name in ((10, "hot idle"), (11, "cold idle")):
    m = ph == p
    if m.any():
        print(f"{name} CTAs {m.sum()}")
