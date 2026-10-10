"""Consumer waits per unit (hot CTAs, last call): true landing latency of the
units the consumer waited on, wait size distribution, and the lead each
waited unit had (issue -> consumer arrival).   wait_dist.py t.pt [label]"""
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
g = np.nonzero(np.diff(st) > 20_000)[0]
t0 = st[g[-1] + 1] if len(g) else st[0]
u = r[(ph >= 120) & (ph < 124) & (r[:, 2] >= t0)]
kind = (u[:, 0] & 0xFF) - 120
q = (u[:, 0] >> 48) & 0xFFFF
iss, wst, full = [(u[:, i] - t0) / 1e3 for i in (1, 2, 3)]
wait = full - wst
for k, n in ((1, "R0"), (3, "R1")):
    m = (kind == k) & (q == 0) & (full > 14)  # past the first fill
    w = wait[m]
    waited = w > 0.05
    lat = (full - iss)[m][waited]
    lead = (wst - iss)[m][waited]
    print(f"{n}: units {m.sum()}, waited {waited.mean():.2f}; wait sum by size: "
          + " ".join(f"<{b}:{w[(w > a) & (w <= b)].sum() / w[w > 0].sum():.2f}"
                     for a, b in ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 99))))
    print(f"    waited units: landing latency p10 {np.percentile(lat, 10):.2f} p50 {np.median(lat):.2f} "
          f"p90 {np.percentile(lat, 90):.2f}; lead (issue->arrival) p50 {np.median(lead):.2f}; "
          f"wait p50 {np.median(w[waited]):.2f}")
