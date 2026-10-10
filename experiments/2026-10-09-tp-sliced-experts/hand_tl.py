"""Warp 0's per-unit work split (v35 TD_UNIT_TRACE), hot CTAs, last call:
routed consume (math), w2 flush, w13 flush, handoff sync, done count, and
activation + ready release. us per CTA and per event.   hand_tl.py t.pt"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
g = np.nonzero(np.diff(st) > 20_000)[0]
t0 = st[g[-1] + 1] if len(g) else st[0]
blk = (r[:, 0] >> 8) & 0xFF
hot = (blk >= 16) & (r[:, 1] >= t0)
ncta = len(np.unique(blk[hot & (ph == 151)]))
rows = []


def stat(name, v):
    if len(v):
        rows.append(f"  {name:28s} per CTA {v.sum() / ncta:6.2f}   n/CTA {len(v) / ncta:5.1f}   "
                    f"p50 {np.median(v):5.2f} p90 {np.percentile(v, 90):5.2f} max {v.max():6.2f}")


d = lambda m, a, b: (r[m, b] - r[m, a]) / 1e3  # noqa: E731
stat("R0 consume (math)", d(hot & (ph == 151), 1, 2))
stat("R1 consume (math)", d(hot & (ph == 153), 1, 2))
stat("R1 flush", d(hot & (ph == 153), 2, 3))
stat("R0 flush (w13 last chunk)", d(hot & (ph == 140), 1, 2))
stat("handoff bar.sync wait", d(hot & (ph == 140), 2, 3))
m = hot & ((ph == 141) | (ph == 142))
stat("done count (fence+atomic)", d(m, 1, 2))
stat("activate + release (done)", d(hot & (ph == 142), 2, 3))
stat("not done tail", d(hot & (ph == 141), 2, 3))
print(f"hot CTAs {ncta}")
print("\n".join(rows))
m = hot & (ph == 151)
v = (r[m, 1] - r[m, 3]) / 1e3
print(f"  R0 full -> math start         per CTA {v.sum() / ncta:6.2f}   p50 {np.median(v):5.2f} p90 {np.percentile(v, 90):5.2f}")
m = hot & (ph == 154)
v = (r[m, 2] - r[m, 1]) / 1e3
if len(v):
    print(f"  R1 full -> math start         per CTA {v.sum() / ncta:6.2f}   p50 {np.median(v):5.2f} p90 {np.percentile(v, 90):5.2f}")
# per-unit cycle: consecutive R0 full stamps on one CTA (non-last units)
m = hot & (ph == 151)
o = np.lexsort((r[m, 3], blk[m]))
f, b = r[m, 3][o], blk[m][o]
dd = np.diff(f)[b[1:] == b[:-1]] / 1e3
print(f"  R0 full -> next R0 full       p10 {np.percentile(dd, 10):5.2f} p50 {np.median(dd):5.2f} p90 {np.percentile(dd, 90):5.2f}")
