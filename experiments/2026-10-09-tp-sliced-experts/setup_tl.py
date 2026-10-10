"""The producer's group setup, split by v43's stamps (hot CTAs, R1 groups):
170 group start -> 174.sa (claim issue) -> 174.sb (index math) -> 174.sc
(entry record in registers) [-> 175 prefetch issued] -> 171.t0 (empty-wait
start).   setup_tl.py t.pt"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
g = np.nonzero(np.diff(st) > 20_000)[0]
t0 = st[g[-1] + 1] if len(g) else st[0]
blk = (r[:, 0] >> 8) & 0xFF
hot = set(np.unique(blk[(ph == 0) & (r[:, 3] >= t0)]))
key = lambda e: e[1] if (e[0] & 0xFF) != 174 else e[1]  # noqa: E731
seg = {k: [] for k in ("start->claim issued", "index math", "record wait",
                       "prefetch issue", "-> empty wait")}
for b in hot:
    m = (blk == b) & (r[:, 1] >= t0) & np.isin(ph, (170, 171, 174, 175))
    ev = r[m][np.argsort(r[m][:, 1], kind="stable")]
    p = ev[:, 0] & 0xFF
    for i in range(len(ev) - 3):
        if p[i] != 170 or p[i + 1] != 174:
            continue
        j = i + 2
        pf = None
        if p[j] == 175:
            pf = ev[j]
            j += 1
        if p[j] != 171 or ((ev[j, 0] >> 32) & 0xFFFF) != 3:
            continue
        a, s = ev[i], ev[i + 1]
        seg["start->claim issued"].append(s[1] - a[1])
        seg["index math"].append(s[2] - s[1])
        seg["record wait"].append(s[3] - s[2])
        seg["prefetch issue"].append(0 if pf is None else pf[1] - s[3])
        end = s[3] if pf is None else pf[1]
        seg["-> empty wait"].append(ev[j, 1] - end)
n = len(seg["index math"])
print(f"R1 groups {n}: mean / p50 / p90 us")
tot = 0
for k, v in seg.items():
    v = np.array(v) / 1e3
    tot += v.mean()
    print(f"  {k:20s} {v.mean():5.2f} {np.median(v):5.2f} {np.percentile(v, 90):5.2f}")
print(f"  {'sum':20s} {tot:5.2f}")
