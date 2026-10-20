"""Where the equal-hot-set rest goes (launch_eq.sh): per M=16 verify step on
device 0, kernel time summed by kernel name and launches per step; median
per arm and hold, arm - x per hold, mean over holds (us). Kernel time on a
side stream lengthens the step only when the main stream waits for it, and
the base already runs several streams, so this attributes, it doesn't sum
to the span difference.

    ak_eqdiff.py [arms] [win]   (default yz, both windows)
"""
import collections
import sqlite3
import statistics as st
import sys
from pathlib import Path

R = Path("/e/fscratch/profound/naeimitabiei1/sliced-prof")
arms = sys.argv[1] if len(sys.argv) > 1 else "yz"
wins = [int(sys.argv[2])] if len(sys.argv) > 2 else [1, 2]


def per_step(db):
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    per = collections.defaultdict(list)
    for s, e, n, cid, sid, gx in c.execute(
            "select start, end, shortName, correlationId, streamId, gridX from "
            "CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0"):
        per[cid].append((s, e, names[n], sid, gx))
    out = []
    for v in per.values():
        if len(v) < 500:
            continue
        m = [k[4] for k in v if k[2].startswith("cooperative_topk")]
        if not m or m[0] != 16:
            continue
        d = collections.Counter()
        for s, e, n, _, _ in v:
            d[("us", n)] += (e - s) / 1e3
            d[("n", n)] += 1
        out.append(d)
    return out


for win in wins:
    diffs = {a: collections.defaultdict(list) for a in arms}
    base_med = collections.defaultdict(list)
    for dx in sorted(R.glob("akx-dec4-24248*")):
        hold = dx.name.split("-")[-1]
        X = per_step(dx / f"nsys.{win}.sqlite")
        keys = set().union(*X)
        for k in keys:
            base_med[k].append(st.median(s[k] for s in X))
        for a in arms:
            Y = per_step(R / f"ak{a}-dec4-{hold}" / f"nsys.{win}.sqlite")
            for k in keys | set().union(*Y):
                diffs[a][k].append(st.median(s[k] for s in Y) - st.median(s[k] for s in X))
    for a in arms:
        print(f"== {a} - x, window {win}, per step")
        us = {k: v for k, v in diffs[a].items() if k[0] == "us"}
        for k, v in sorted(us.items(), key=lambda t: -abs(st.mean(t[1])))[:14]:
            b = st.mean(base_med[k]) if k in base_med else 0.0
            dn = st.mean(diffs[a][("n", k[1])])
            print(f"  {st.mean(v):+8.1f} us ({min(v):+7.1f} to {max(v):+7.1f})  x {b:8.1f}  "
                  f"launches {dn:+5.0f}  {k[1][:60]}")
