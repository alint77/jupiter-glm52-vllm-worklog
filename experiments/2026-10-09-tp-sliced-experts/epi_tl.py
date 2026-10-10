"""v33 epilogue warp events (last call): per event, barrier wait (140: sync
start -> done) and processing of completing events (141), and consumer posts.
    epi_tl.py t.pt"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
t0 = st[np.nonzero(np.diff(st) > 20_000)[0][-1] + 1] if (np.diff(st) > 20_000).any() else st[0]
blk = (r[:, 0] >> 8) & 0xFF
last = r[:, 3] >= t0
e = r[(ph == 140) & last]
w = (e[:, 2] - e[:, 1]) / 1e3
print(f"events {len(e)}: barrier wait p50 {np.median(w):.2f} p90 {np.percentile(w, 90):.2f} max {w.max():.2f}")
d = r[(ph == 141) & last]
pt = (d[:, 2] - d[:, 1]) / 1e3
print(f"completing {len(d)}: processing p50 {np.median(pt):.2f} p90 {np.percentile(pt, 90):.2f} max {pt.max():.2f}")
# skew: per (block, gen) first and last consumer post
p = r[((ph == 142) | (ph == 143)) & last]
key = p[:, 2] + (((p[:, 0] >> 8) & 0xFF) << 20)
sk = []
for k in np.unique(key):
    t = p[key == k, 1]
    sk.append((t.max() - t.min()) / 1e3)
sk = np.array(sk)
print(f"post skew across warps per event: p50 {np.median(sk):.2f} p90 {np.percentile(sk, 90):.2f} max {sk.max():.2f}")
