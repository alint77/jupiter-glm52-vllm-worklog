"""Per-tier timeline of one traced call (TD_UNIT_TRACE TD_CTA_TRACE): when the
hot and the cold units are consumed, who consumes them, and what the kernel
does after the last hot unit.

    tier_tl.py t.pt [bin_us]
"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
bin_us = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
gaps = np.nonzero(np.diff(st) > 20_000)[0]
t0 = st[gaps[-1] + 1] if len(gaps) else st[0]
us = lambda x: (x - t0) / 1e3  # noqa: E731
blk = (r[:, 0] >> 8) & 0xFF
late = r[:, 3] >= t0
own = {}
ex = {}
for o in (0, 1):
    m = (ph == o) & late
    for b, e in zip(blk[m], r[m, 3]):
        own[b], ex[b] = o, us(e)
end = max(ex.values())
m = (ph >= 120) & (ph < 124) & (r[:, 2] >= t0)
kind = ph[m] - 120
tier = (r[m, 0] >> 48) & 0xFFFF
ub = blk[m]
iss, wst, full = us(r[m, 1]), us(r[m, 2]), us(r[m, 3])
names = ["S0", "R0", "S1", "R1"]
cold_ctas = sum(own.values())
print(f"kernel end {end:6.1f} us; CTAs {len(own)} ({cold_ctas} cold)")
for q in (0, 1):
    for k in range(4):
        s = (tier == q) & (kind == k)
        if not s.any():
            continue
        byo = [int(((np.array([own[b] for b in ub[s]])) == o).sum()) for o in (0, 1)]
        print(f"  {'hot ' if q == 0 else 'cold'} {names[k]}: {s.sum():4d} units  "
              f"first issue {iss[s].min():6.1f}  last full {full[s].max():6.1f}  "
              f"by hot/cold CTAs {byo[0]}/{byo[1]}")
hx = [e for b, e in ex.items() if own[b] == 0]
cx = [e for b, e in ex.items() if own[b] == 1]
print(f"  hot CTAs exit p50 {np.median(hx):6.1f} max {max(hx):6.1f}; "
      f"cold CTAs exit p50 {np.median(cx):6.1f} max {max(cx):6.1f}")
nb = int(np.ceil(end / bin_us))
print(f"units landing per {bin_us:g} us (hot R0/R1, cold R0/R1), active CTAs hot/cold")
for i in range(nb):
    lo, hi = i * bin_us, (i + 1) * bin_us
    c = [int(((tier == q) & (kind == k) & (full >= lo) & (full < hi)).sum())
         for q in (0, 1) for k in (1, 3)]
    ah = sum(1 for b, e in ex.items() if own[b] == 0 and e > lo)
    ac = sum(1 for b, e in ex.items() if own[b] == 1 and e > lo)
    print(f"  {lo:5.0f}-{hi:4.0f}  hot {c[0]:3d}/{c[1]:3d}  cold {c[2]:3d}/{c[3]:3d}  CTAs {ah:3d}/{ac:2d}")
