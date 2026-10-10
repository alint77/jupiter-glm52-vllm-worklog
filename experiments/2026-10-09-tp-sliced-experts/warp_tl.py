"""Per consumer warp: total wait on full stages and exit time (last call),
hot CTAs; and the producer's spin / empty-wait totals.   warp_tl.py t.pt"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
t0 = st[np.nonzero(np.diff(st) > 20_000)[0][-1] + 1] if (np.diff(st) > 20_000).any() else st[0]
blk = (r[:, 0] >> 8) & 0xFF
for w in range(8):
    m = (ph == 130 + w) & (r[:, 1] >= t0) & (blk >= 16)
    wait = (r[m, 2] - r[m, 1]) / 1e3
    ex = (r[m, 3] - t0) / 1e3
    print(f"warp {w}: wait p50 {np.median(wait):5.1f} p10 {np.percentile(wait, 10):5.1f} p90 {np.percentile(wait, 90):5.1f}  exit p50 {np.median(ex):5.1f}")
for nm, base in (("producer ready-spin", 60), ("producer empty-wait", 80)):
    m = (ph == base) & (r[:, 1] >= t0) & (blk >= 16)
    v = (r[m, 2] - r[m, 1]) / 1e3
    print(f"{nm} (hot): p50 {np.median(v):5.1f} p90 {np.percentile(v, 90):5.1f}")
