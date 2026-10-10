"""The producer's time per hot CTA (v39 TD_UNIT_TRACE, one traced call):
empty waits, ready spins, claim waits, group setup and issue work, by the
kind of group it is serving.   prod_tl.py t.pt"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
g = np.nonzero(np.diff(st) > 20_000)[0]
t0 = st[g[-1] + 1] if len(g) else st[0]
blk = (r[:, 0] >> 8) & 0xFF
hot = {b for b in np.unique(blk[(ph == 0) & (r[:, 3] >= t0)])}
names = ["S0", "R0", "S1", "R1"]
tot = {}
ncta = 0
for b in np.unique(blk[(ph == 171) & (r[:, 1] >= t0)]):
    if b not in hot:
        continue
    ncta += 1
    ev = []
    for p in (170, 171, 172, 173):
        m = (ph == p) & (blk == b) & (r[:, 1] >= t0)
        for row in r[m]:
            ev.append((row[1], p, row))
    ev.sort(key=lambda e: e[0])
    kind = 1
    prev_end = None
    for i, (t, p, row) in enumerate(ev):
        if p == 171:
            kind = (row[0] >> 32) & 0xFFFF
            key = names[kind]
            tot[("empty wait", key)] = tot.get(("empty wait", key), 0) + (row[2] - row[1])
        elif p == 172:
            tot[("ready spin", "R1")] = tot.get(("ready spin", "R1"), 0) + (row[2] - row[1])
        elif p == 173:
            tot[("claim wait", names[kind])] = tot.get(("claim wait", names[kind]), 0) + (row[2] - row[1])
    # issue / setup: span between events not covered by waits
print(f"hot CTAs {ncta}; producer us per CTA")
for (a, k), v in sorted(tot.items()):
    print(f"  {a:12s} {k}: {v / ncta / 1e3:6.2f}")
m = (ph == 171) & (r[:, 1] >= t0) & np.isin(blk, list(hot))
kind = (r[m, 0] >> 32) & 0xFFFF
for k in (1, 3):
    mm = kind == k
    o = np.lexsort((r[m][mm, 1], blk[m][mm]))
    tt, bb = r[m][mm, 2][o], blk[m][mm][o]
    t_next = r[m][mm, 1][o]
    gaps = (t_next[1:] - tt[:-1])[bb[1:] == bb[:-1]] / 1e3
    print(f"  {names[k]}: issue -> next unit's empty-wait start p50 {np.median(gaps):.2f} p90 {np.percentile(gaps, 90):.2f} us"
          f" (units/CTA {mm.sum() / ncta:.1f})")

# the producer's path per R1 group (GR1=1: one unit), in order:
#   170 group start -> 171.t0 empty-wait start   claim issue, group_at, record
#   171 empty wait
#   171.t1 -> 172.t0   descriptors + weight TMAs
#   172 ready spin (only when the entry changes)
#   172.t1 (or 171.t1) -> 173.t0   x row copies, loop end
#   173 claim wait;  173.t1 -> next 170   loop head
seg = {k: [] for k in ("setup+record", "empty wait", "W issue", "ready spin",
                       "x copies", "claim wait", "loop head")}
for b in hot:
    m = (blk == b) & (r[:, 1] >= t0) & np.isin(ph, (170, 171, 172, 173))
    ev = r[m][np.argsort(r[m][:, 1], kind="stable")]
    p_ = ev[:, 0] & 0xFF
    for i in range(len(ev) - 5):
        if p_[i] != 170 or p_[i + 1] != 171 or ((ev[i + 1, 0] >> 32) & 0xFFFF) != 3:
            continue
        j = i + 2
        sp = None
        if p_[j] == 172:
            sp = ev[j]
            j += 1
        if p_[j] != 173 or j + 1 >= len(ev) or p_[j + 1] != 170:
            continue
        a, u, c, nx = ev[i], ev[i + 1], ev[j], ev[j + 1]
        seg["setup+record"].append(u[1] - a[1])
        seg["empty wait"].append(u[2] - u[1])
        if sp is not None:
            seg["W issue"].append(sp[1] - u[2])
            seg["ready spin"].append(sp[2] - sp[1])
            seg["x copies"].append(c[1] - sp[2])
        else:
            seg["W issue"].append(0)
            seg["ready spin"].append(0)
            seg["x copies"].append(c[1] - u[2])
        seg["claim wait"].append(c[2] - c[1])
        seg["loop head"].append(nx[1] - c[2])
n = len(seg["loop head"])
print(f"R1 single-unit groups {n}: mean / p50 / p90 us"
      f" (ready spin taken in {np.mean(np.array(seg['W issue']) > 0):.0%})")
tot = 0
for k, v in seg.items():
    v = np.array(v) / 1e3
    tot += v.mean()
    print(f"  {k:14s} {v.mean():5.2f} {np.median(v):5.2f} {np.percentile(v, 90):5.2f}")
print(f"  {'sum':14s} {tot:5.2f}")
