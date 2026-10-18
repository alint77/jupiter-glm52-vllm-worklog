"""ABBA hot-order A/B (launch_ab.sh): per hold, arm b minus arm a on the same
node. Verify steps = CUDA graph replays with >= 500 kernels (device 0); M =
the indexer top-k grid's x (tokens in the step). Per step: GPU span of the
graph (first kernel start to last end) and the sum of its 75 layer_kernel
calls. Grouped by kind, window (dec arms: 1 = ~5K, 2 = ~50K context) and M;
medians per arm, then b - a per hold and the mean over holds.

    ab_compare.py [sliced-prof root]
"""
import collections
import sqlite3
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/e/fscratch/profound/naeimitabiei1/sliced-prof")


def steps(db):
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    ks = c.execute("select start, end, shortName, correlationId, gridX from "
                   "CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0").fetchall()
    per = collections.defaultdict(list)
    for k in ks:
        per[k[3]].append(k)
    out = []
    for v in per.values():
        if len(v) < 500:
            continue
        m = [k[4] for k in v if names[k[2]].startswith("cooperative_topk")]
        moe = sum(k[1] - k[0] for k in v if names[k[2]] == "layer_kernel") / 1e6
        out.append((m[0] if m else 0, (max(k[1] for k in v) - min(k[0] for k in v)) / 1e6, moe))
    return out


res = collections.defaultdict(dict)  # (kind, win, M) -> {(hold, arm): (n, span, moe)}
for d in sorted(root.glob("ho[ab]-*")):
    arm, kind, hold = d.name[2], d.name.split("-")[1], d.name.split("-")[2]
    for db in sorted(d.glob("nsys.*.sqlite")):
        win = int(db.name.split(".")[1]) if kind.startswith("dec") else 0
        by = collections.defaultdict(list)
        for m, span, moe in steps(db):
            by[m].append((span, moe))
        for m, v in by.items():
            key = (kind, win, m)
            old = res[key].get((hold, arm), [])
            res[key][(hold, arm)] = old + v
print(f"{'kind':7s} {'win':>3s} {'M':>3s} | {'hold':8s} {'n a/b':>9s} {'span a':>7s} {'b-a':>6s} "
      f"{'moe a':>6s} {'b-a':>6s}  (ms, medians)")
for key in sorted(res):
    holds = sorted({h for h, _ in res[key]})
    ds, dm = [], []
    for h in holds:
        a, b = res[key].get((h, "a")), res[key].get((h, "b"))
        if not a or not b or min(len(a), len(b)) < 15:
            continue
        sa, sb = st.median(x[0] for x in a), st.median(x[0] for x in b)
        ma, mb = st.median(x[1] for x in a), st.median(x[1] for x in b)
        ds.append(sb - sa)
        dm.append(mb - ma)
        print(f"{key[0]:7s} {key[1]:3d} {key[2]:3d} | {h:8s} {len(a):4d}/{len(b):<4d} {sa:7.2f} "
              f"{sb - sa:+6.2f} {ma:6.2f} {mb - ma:+6.2f}")
    if len(ds) > 1:
        print(f"{'':15s} | mean over {len(ds)} holds: span {st.mean(ds):+.2f}  moe {st.mean(dm):+.2f}")
