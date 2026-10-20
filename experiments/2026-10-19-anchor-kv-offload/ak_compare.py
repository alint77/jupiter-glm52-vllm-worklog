"""Anchor-offload ABC (launch_abc.sh): arms b and c minus arm a on the same
hold. Verify steps = CUDA graph replays with >= 500 kernels on device 0; M =
the cooperative top-k grid's x. Per step: graph GPU span, rest = span minus
the 75 layer_kernel calls (MoE swings with the cold experts the traffic
touches), the anchor map kernels (sum, calls) and the sparse FlashMLA decode
kernels. Medians per arm, x - a per hold, mean over holds; then kernels whose
per-step launch count differs from arm a.

    ak_compare.py [sliced-prof root]
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
        mp = [d for d, n in zip(dur, nm) if n.startswith("map_kernel")]
        fm = sum(d for d, n in zip(dur, nm) if "sparse" in n.lower() and "decode" in n.lower())
        moe = sum(d for d, n in zip(dur, nm) if n == "layer_kernel")
        span = (max(k[1] for k in v) - min(k[0] for k in v)) / 1e3
        out.append(dict(M=m[0] if m else 0, span=span, rest=span - moe, map=sum(mp),
                        nmap=len(mp), fmla=fm, names=collections.Counter(nm)))
    return out


res = collections.defaultdict(lambda: collections.defaultdict(list))
for d in sorted(root.glob("ak[abcst]-*")):
    arm, kind, hold = d.name[2], d.name.split("-")[1], d.name.split("-")[2]
    for db in sorted(d.glob("nsys.*.sqlite")):
        win = int(db.name.split(".")[1])
        for s in steps(db):
            res[(kind, win, s["M"])][(hold, arm)].append(s)

print(f"{'kind':6s} {'win':>3s} {'M':>3s} | {'hold':8s} arm {'n a/x':>9s} {'span a':>7s} {'x-a':>6s} "
      f"{'rest x-a':>8s} {'map x':>6s} {'calls':>5s} {'fmla x-a':>8s}  (ms span/rest, us map/fmla)")
for key in sorted(res):
    holds = sorted({h for h, _ in res[key]})
    for arm in "bcst":
        agg = collections.defaultdict(list)
        for h in holds:
            a, b = res[key].get((h, "a")), res[key].get((h, arm))
            if not a or not b or min(len(a), len(b)) < 15:
                continue
            med = lambda xs, f: st.median(x[f] for x in xs)
            d = {f: med(b, f) - med(a, f) for f in ("span", "rest", "fmla")}
            for f, v in d.items():
                agg[f].append(v)
            print(f"{key[0]:6s} {key[1]:3d} {key[2]:3d} | {h:8s}  {arm}  {len(a):4d}/{len(b):<4d} "
                  f"{med(a, 'span') / 1e3:7.2f} {d['span'] / 1e3:+6.3f} {d['rest'] / 1e3:+8.3f} "
                  f"{med(b, 'map'):6.1f} {b[0]['nmap']:5d} {d['fmla']:+8.1f}")
            ca = sum((x["names"] for x in a), collections.Counter())
            cb = sum((x["names"] for x in b), collections.Counter())
            diff = {n: cb[n] / len(b) - ca[n] / len(a) for n in set(ca) | set(cb)}
            for n, dv in sorted(diff.items(), key=lambda t: -abs(t[1]))[:4]:
                if abs(dv) >= 0.5:
                    print(f"{'':22s} launches/step {arm}-a {dv:+6.1f}  {n[:80]}")
        if len(agg["span"]) > 1:
            print(f"{'':16s} | {arm}: mean over {len(agg['span'])} holds: span "
                  f"{st.mean(agg['span']) / 1e3:+.3f} ms  rest {st.mean(agg['rest']) / 1e3:+.3f} ms  "
                  f"fmla {st.mean(agg['fmla']):+.1f} us")
