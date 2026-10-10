"""Per-unit load timeline of one traced call (v32 TD_UNIT_TRACE): issue ->
stage-full latency per kind, and bytes landing per 2 us bin (hot / cold).

    unit_tl.py t.pt [bin_us]
"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
bin_us = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
ph = r[:, 0] & 0xFF
# the last traced call: from the last cluster of route_prep entries
st = np.sort(r[ph == 50, 1])
t0 = st[np.nonzero(np.diff(st) > 20_000)[0][-1] + 1] if (np.diff(st) > 20_000).any() else st[0]
u = r[(ph >= 120) & (ph < 124) & (r[:, 2] >= t0)]
kind = (u[:, 0] & 0xFF) - 120
ntok = (u[:, 0] >> 32) & 0xFFFF
q = (u[:, 0] >> 48) & 0xFFFF
# word 1: producer issue, word 2: consumer wait start, word 3: stage full
iss, wst, full = (u[:, 1] - t0) / 1e3, (u[:, 2] - t0) / 1e3, (u[:, 3] - t0) / 1e3
blk = (u[:, 0] >> 8) & 0xFF
# per CTA in order: consume time of a unit = next wait start - its full time
o = np.lexsort((full, blk))
cons = np.full(len(u), np.nan)
same = blk[o][1:] == blk[o][:-1]
cons[o[:-1][same]] = wst[o][1:][same] - full[o][:-1][same]
wait = full - wst
W, S = 32768, 4096
nbytes = np.where(kind == 1, W + S + ntok * 1024, np.where(kind == 3, W + S + ntok * 1040, W + 8 * 128))
names = ["S0", "R0", "S1", "R1"]
print(f"units {len(u)}  MB {nbytes.sum() / 1e6:.1f}  first full {full.min():.1f} last full {full.max():.1f} us")
for k in range(4):
    m = kind == k
    if m.any():
        lat = full[m] - iss[m]
        print(f"{names[k]} n {m.sum():5d} latency p10 {np.percentile(lat, 10):5.2f} p50 {np.median(lat):5.2f} "
              f"p90 {np.percentile(lat, 90):5.2f} max {lat.max():6.2f}  full span {full[m].min():5.1f}-{full[m].max():5.1f}")
        c = cons[m][~np.isnan(cons[m])]
        print(f"   consume p10 {np.percentile(c, 10):5.2f} p50 {np.median(c):5.2f} p90 {np.percentile(c, 90):5.2f}"
              f"  sum/CTA {c.sum() / len(np.unique(blk)):5.1f}   wait-before sum/CTA {wait[m].sum() / len(np.unique(blk)):5.1f}")
edges = np.arange(0, full.max() + bin_us, bin_us)
print(f"\n{'t us':>6} {'hot GB/s':>9} {'cold GB/s':>9}  kinds landing (S0 R0 S1 R1)")
for a, b in zip(edges[:-1], edges[1:]):
    m = (full >= a) & (full < b)
    hot = nbytes[m & (q == 0)].sum() / (b - a) / 1e3
    cold = nbytes[m & (q == 1)].sum() / (b - a) / 1e3
    ks = " ".join(f"{(m & (kind == k)).sum():3d}" for k in range(4))
    print(f"{a:6.1f} {hot:9.0f} {cold:9.0f}  {ks}")
