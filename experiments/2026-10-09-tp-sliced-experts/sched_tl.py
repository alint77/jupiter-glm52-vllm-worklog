"""v44 scheduler warp vs producer (hot CTAs, one traced call): per group the
scheduler's claim round trip, decode + record + publish, its wait for a free
FIFO slot, and the producer's wait for the next group (by phase: groups that
start before / after the first R1 issue).   sched_tl.py t.pt"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
g = np.nonzero(np.diff(st) > 20_000)[0]
t0 = st[g[-1] + 1] if len(g) else st[0]
blk = (r[:, 0] >> 8) & 0xFF
hot = np.isin(blk, np.unique(blk[(ph == 0) & (r[:, 3] >= t0)]))
late = r[:, 1] >= t0
names = ["S0", "R0", "S1", "R1"]
ncta = len(np.unique(blk[hot & (ph == 176) & late]))
print(f"hot CTAs {ncta}")
m = hot & late & (ph == 176)
kind = (r[m, 0] >> 32) & 0xFFFF
for k in range(4):
    mm = kind == k
    if not mm.any():
        continue
    claim = (r[m][mm, 2] - r[m][mm, 1]) / 1e3
    rest = (r[m][mm, 3] - r[m][mm, 2]) / 1e3
    rdy = ((r[m][mm, 0] >> 48) & 0xFFFF).mean()
    print(f"  scheduler {names[k]}: ready at probe {rdy:.0%}  groups/CTA {mm.sum() / ncta:5.1f}  claim RT p50 {np.median(claim):.2f}"
          f" p90 {np.percentile(claim, 90):.2f}  decode+record+publish p50 {np.median(rest):.2f}"
          f" p90 {np.percentile(rest, 90):.2f} us")
m = hot & late & (ph == 177)
w = (r[m, 2] - r[m, 1]) / 1e3
print(f"  scheduler slot wait: per CTA {w.sum() / ncta:5.2f} us, p50 {np.median(w):.2f}")
m = hot & late & (ph == 173)
w = (r[m, 2] - r[m, 1]) / 1e3
r1s = {}
mi = hot & late & (ph == 171) & (((r[:, 0] >> 32) & 0xFFFF) == 3)
for b, t in zip(blk[mi], r[mi, 1]):
    r1s[b] = min(r1s.get(b, 1 << 62), t)
in_r1 = np.array([t >= r1s.get(b, 1 << 62) for b, t in zip(blk[m], r[m, 1])])
for lab, sel in (("w13 phase", ~in_r1), ("w2 phase", in_r1)):
    print(f"  producer wait for next group, {lab}: per CTA {w[sel].sum() / ncta:5.2f} us,"
          f" n/CTA {sel.sum() / ncta:5.1f}, p50 {np.median(w[sel]) if sel.any() else 0:.2f}")
