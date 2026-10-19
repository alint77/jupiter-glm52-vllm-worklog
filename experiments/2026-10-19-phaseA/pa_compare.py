"""Phase A ABBA (launch_ab.sh): arm n minus arm a on the same hold. Steps =
CUDA graph replays with >= 500 kernels on device 0; M = the cooperative top-k
grid's x. Per step: graph GPU span, paged MQA logits kernel time (sum, and
calls) and the sum of the 75 layer_kernel calls (MoE swings with the cold
experts the traffic touches, so span - MoE is the comparable rest). Medians per
arm, n - a per hold, mean over holds; then the kernels whose per-step count
differs between arms (new or removed launches).

    pa_compare.py [sliced-prof root]
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
        nm = [names[k[2]] for k in v]
        m = [k[4] for k, n in zip(v, nm) if n.startswith("cooperative_topk")]
        dur = [(k[1] - k[0]) / 1e3 for k in v]
        lg = [d for d, n in zip(dur, nm) if "paged_mqa_logits" in n and "metadata" not in n]
        moe = sum(d for d, n in zip(dur, nm) if n == "layer_kernel")
        span = (max(k[1] for k in v) - min(k[0] for k in v)) / 1e3
        out.append((m[0] if m else 0, span, sum(lg), len(lg), moe, collections.Counter(nm)))
    return out


res = collections.defaultdict(lambda: collections.defaultdict(list))
for d in sorted(root.glob("pa[an]-*")):
    arm, kind, hold = d.name[2], d.name.split("-")[1], d.name.split("-")[2]
    for db in sorted(d.glob("nsys.*.sqlite")):
        win = int(db.name.split(".")[1])
        for s in steps(db):
            res[(kind, win, s[0])][(hold, arm)].append(s)

print(f"{'kind':5s} {'win':>3s} {'M':>3s} | {'hold':8s} {'n a/n':>9s} {'span a':>7s} {'n-a':>6s} "
      f"{'rest n-a':>8s} {'logits a':>8s} {'n-a':>7s} {'calls':>5s}  (ms span/rest, us logits)")
for key in sorted(res):
    holds = sorted({h for h, _ in res[key]})
    agg = collections.defaultdict(list)
    for h in holds:
        a, b = res[key].get((h, "a")), res[key].get((h, "n"))
        if not a or not b or min(len(a), len(b)) < 15:
            continue
        med = lambda xs, f: st.median(f(x) for x in xs)
        sa, sb = med(a, lambda x: x[1]), med(b, lambda x: x[1])
        ra, rb = med(a, lambda x: x[1] - x[4]), med(b, lambda x: x[1] - x[4])
        la, lb = med(a, lambda x: x[2]), med(b, lambda x: x[2])
        agg["span"].append(sb - sa)
        agg["rest"].append(rb - ra)
        agg["logits"].append(lb - la)
        print(f"{key[0]:5s} {key[1]:3d} {key[2]:3d} | {h:8s} {len(a):4d}/{len(b):<4d} {sa / 1e3:7.2f} "
              f"{(sb - sa) / 1e3:+6.3f} {(rb - ra) / 1e3:+8.3f} {la:8.1f} {lb - la:+7.1f} "
              f"{a[0][3]:2d}/{b[0][3]:<2d}")
        ca = sum((x[5] for x in a), collections.Counter())
        cb = sum((x[5] for x in b), collections.Counter())
        diff = {n: cb[n] / len(b) - ca[n] / len(a) for n in set(ca) | set(cb)}
        for n, dv in sorted(diff.items(), key=lambda t: -abs(t[1]))[:4]:
            if abs(dv) >= 0.5:
                print(f"{'':17s} launches/step n-a {dv:+6.1f}  {n[:80]}")
    if len(agg["span"]) > 1:
        print(f"{'':15s} | mean over {len(agg['span'])} holds: span {st.mean(agg['span']) / 1e3:+.3f} ms  "
              f"rest {st.mean(agg['rest']) / 1e3:+.3f} ms  logits {st.mean(agg['logits']):+.1f} us")
