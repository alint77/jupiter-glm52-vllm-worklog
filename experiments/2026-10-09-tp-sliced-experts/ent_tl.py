"""Per-entry timeline (v52 trace records), last call, us from route_prep start:
last w13 group claimed, done count started (the finisher got the entry's last
handoff), ready released, first w2 claim, w2 claims published not-ready, and
producer ready-spin time on the entry.   ent_tl.py t.pt [q]"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
qsel = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
g = np.nonzero(np.diff(st) > 20_000)[0]
t0 = st[g[-1] + 1] if len(g) else st[0]
r = r[r[:, 1] >= t0]
ph = r[:, 0] & 0xFF
nh = (r[:, 0] >> 32) & 0xFFFF
nc = (r[:, 0] >> 48) & 0xFFFF
us = lambda t: (t - t0) / 1e3  # noqa: E731
s = ph == 176
kind, sei, sq, srdy = nh[s] & 0xF, nh[s] >> 4, nc[s] >> 1, nc[s] & 1
sclaim = r[s, 2]
d = ph == 142
dei, dq = nh[d] & 0xFFF, nh[d] >> 12
sp = ph == 172
pei, pq = nh[sp] & 0xFFF, nh[sp] >> 12
rows = []
for ei, q, hc, rel in zip(dei, dq, r[d, 1], r[d, 3]):
    if q != qsel:
        continue
    m0 = (kind == 1) & (sei == ei) & (sq == q)
    m1 = (kind == 3) & (sei == ei) & (sq == q)
    mp = (pei == ei) & (pq == q)
    rows.append((us(rel), ei, us(sclaim[m0].max()) if m0.any() else np.nan, us(hc),
                 us(sclaim[m1].min()) if m1.any() else np.nan,
                 us(sclaim[m1].max()) if m1.any() else np.nan,
                 int(m1.sum()), int((m1 & (srdy == 0)).sum()),
                 (r[sp][mp, 2] - r[sp][mp, 1]).sum() / 1e3))
rows.sort()
print(f"kernel end {us(r[:, 3].max()):.1f} us; tier q={qsel}, {len(rows)} entries")
print(" entry  lastR0claim  count@  ready@  R1 first  R1 last  R1 n  notready  spin us  slack")
for rel, ei, l0, hc, f1, l1, n1, nr, spin in rows:
    print(f" {ei:5d} {l0:11.1f} {hc:7.1f} {rel:7.1f} {f1:9.1f} {l1:8.1f} {n1:5d} {nr:9d} {spin:8.1f} {f1 - rel:6.1f}")
v = np.array([x[-1] for x in rows])
print(f"total producer spin on tier {qsel}: {v.sum():.1f} us (all CTAs)")
