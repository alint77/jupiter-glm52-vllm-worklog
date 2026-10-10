"""Time breakdown of one traced call (v32+ TD_UNIT_TRACE TD_CTA_TRACE): every
CTA's wall time from route_prep entry to the kernel's last exit, partitioned
into head, consume per kind, waits by cause and tail, plus a per-bin view.

A consumer wait on unit u (full_u - wst_u) is charged to:
  latency  the load was in flight when the consumer arrived (issue < wst)
  ring     issued late because its slot was still held: the issue follows
           the release of unit u - STAGES (~ wst of u - STAGES + 1)
  producer issued late for the producer's own reasons (claim / record /
           ready spin / issue work)
For late issues, issue -> full is charged to latency too ("late latency").

    trace_bd.py t.pt [stages] [bin_us]
"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
STAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 4
bin_us = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
gaps = np.nonzero(np.diff(st) > 20_000)[0]
t0 = st[gaps[-1] + 1] if len(gaps) else st[0]
us = lambda x: (x - t0) / 1e3  # noqa: E731
blk_all = (r[:, 0] >> 8) & 0xFF
late = r[:, 3] >= t0

rp = (ph == 50) & late
print(f"route_prep: exit {us(r[rp, 3].max()):5.1f} us")
cta = {}
for own in (0, 1):
    m = (ph == own) & late
    for b, ready, ex in zip(blk_all[m], r[m, 1], r[m, 3]):
        cta[b] = (own, us(ready), us(ex))
end = max(v[2] for v in cta.values())
print(f"kernel end {end:5.1f} us, CTAs {len(cta)}")

u = r[(ph >= 120) & (ph < 124) & (r[:, 2] >= t0)]
kind = (u[:, 0] & 0xFF) - 120
blk = (u[:, 0] >> 8) & 0xFF
iss, wst, full = us(u[:, 1]), us(u[:, 2]), us(u[:, 3])
names = ["S0", "R0", "S1", "R1"]

cats = ["head", "first fill"] + [f"consume {n}" for n in names] + [
    f"wait {c} {n}" for c in ("latency", "ring", "producer") for n in names
] + ["late latency", "after last unit", "tail (idle to end)"]
acc = {c: [] for c in cats}
nb = int(np.ceil(end / bin_us))
bins = np.zeros((nb, 3))  # consume, wait, idle (CTA-us per bin)


def add_bins(a, b, col):
    for i in range(int(a // bin_us), min(nb, int(np.ceil(b / bin_us)))):
        lo, hi = max(a, i * bin_us), min(b, (i + 1) * bin_us)
        if hi > lo:
            bins[i, col] += hi - lo


for b, (own, ready, ex) in cta.items():
    if own:  # hot CTAs only below; cold CTAs feed C2C
        continue
    m = blk == b
    o = np.argsort(wst[m])
    k, i_, w_, f_ = kind[m][o], iss[m][o], wst[m][o], full[m][o]
    v = {c: 0.0 for c in cats}
    v["head"] = ready
    v["first fill"] = max(0.0, w_[0] - ready) if len(w_) else 0.0
    for j in range(len(k)):
        n = names[k[j]]
        wait = max(0.0, f_[j] - w_[j])
        if wait > 0:
            if i_[j] <= w_[j]:
                v[f"wait latency {n}"] += wait
            else:
                rel = w_[j - STAGES + 1] if j >= STAGES - 1 else -1e9
                late_by = min(i_[j], f_[j]) - w_[j]
                if i_[j] - rel < 0.3:
                    v[f"wait ring {n}"] += late_by
                else:
                    v[f"wait producer {n}"] += late_by
                v["late latency"] += max(0.0, f_[j] - max(i_[j], w_[j]))
        add_bins(w_[j], f_[j], 1)
        nxt = w_[j + 1] if j + 1 < len(k) else ex
        v[f"consume {n}"] += max(0.0, nxt - f_[j]) if j + 1 < len(k) else 0.0
        add_bins(f_[j], nxt, 0)
    if len(k):
        v["after last unit"] = max(0.0, ex - f_[-1])
    v["tail (idle to end)"] = end - ex
    add_bins(ex, end, 2)
    add_bins(0, w_[0] if len(w_) else ex, 2)
    for c in cats:
        acc[c].append(v[c])

nh = len(acc["head"])
print(f"\nhot CTAs {nh}: mean us per CTA (sum = kernel end {end:.1f})")
tot = 0.0
for c in cats:
    a = np.array(acc[c])
    tot += a.mean()
    if a.mean() > 0.05:
        print(f"  {c:22s} {a.mean():6.1f}   p10 {np.percentile(a, 10):6.1f} p90 {np.percentile(a, 90):6.1f}")
print(f"  {'total':22s} {tot:6.1f}")

print(f"\nper {bin_us:g} us bin, hot CTAs: share of CTA time consuming / waiting / idle")
for i in range(nb):
    s = bins[i] / (nh * bin_us)
    print(f"  {i * bin_us:5.0f}-{(i + 1) * bin_us:4.0f}  consume {s[0]:4.2f}  wait {s[1]:4.2f}  idle {s[2]:4.2f}")
