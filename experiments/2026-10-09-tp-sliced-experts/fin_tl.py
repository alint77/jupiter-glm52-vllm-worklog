"""Finisher done-path split (v50 records 142/179/143), all CTAs, last call:
count RT -> post-count fence -> first loads landed -> stores issued ->
proxy+GPU fences -> ready release; also by route count.   fin_tl.py t.pt"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
g = np.nonzero(np.diff(st) > 20_000)[0]
t0 = st[g[-1] + 1] if len(g) else st[0]
r = r[r[:, 1] >= t0]
ph = r[:, 0] & 0xFF
blk = (r[:, 0] >> 8) & 0xFF
a, b = r[ph == 179], r[ph == 143]
d = r[ph == 142]
key = lambda x: ((x[:, 0] >> 8) & 0xFF) * (1 << 40) + x[:, 1]  # noqa: E731
# 142 {hc, hd, end}; 179 {a0, al, a1}; 143 {a1, a2, rel}: pair by CTA + time
a = a[np.argsort(key(a))]
b = b[np.argsort(key(b))]
d = d[np.argsort(key(d))]
assert len(a) == len(b) == len(d), (len(a), len(b), len(d))
n = (a[:, 0] >> 32) & 0xFFFF
cols = {
    "count RT": (d[:, 2] - d[:, 1]),
    "post-count fence": (a[:, 1] - d[:, 2]),
    "loads landed": (a[:, 2] - a[:, 1]),
    "rest to stores": (a[:, 3] - a[:, 2]),
    "fences": (b[:, 2] - b[:, 1]),
    "release": (b[:, 3] - b[:, 2]),
    "total": (b[:, 3] - d[:, 1]),
}
print(f"done events {len(a)}, routes mean {n.mean():.2f} max {n.max()}")
for k, v in cols.items():
    v = v / 1e3
    print(f"  {k:18s} p50 {np.median(v):6.2f} p90 {np.percentile(v, 90):6.2f} mean {v.mean():6.2f}")
for k in np.unique(n):
    m = n == k
    print(f"  n={k}: {m.sum():4d}  " + "  ".join(
        f"{c[:6]} {np.median(v[m]) / 1e3:5.2f}" for c, v in cols.items()))
