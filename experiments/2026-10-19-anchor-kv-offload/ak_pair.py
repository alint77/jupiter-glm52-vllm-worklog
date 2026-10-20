"""Same-node pairs for launch_eq.sh / launch_buy.sh: each arm minus the base
arm on the same hold, verify steps at M=16 (as ak_compare.py: CUDA graph
replays with >= 500 kernels on device 0). Per step: graph GPU span, MoE (the
75 layer_kernel calls), rest = span - MoE, sparse FlashMLA decode. Medians per
arm and window, x - base per hold, mean and range over holds. Writes the pairs
to <prefix>_pairs.json next to this script.

    ak_pair.py <tag prefix> <base arm> <arms> [sliced-prof root]
    ak_pair.py ak x yz      # launch_eq.sh: equal hot set
    ak_pair.py akb x uw     # launch_buy.sh: each arm's own hot set
"""
import collections
import json
import sqlite3
import statistics as st
import sys
from pathlib import Path

prefix, base, arms = sys.argv[1:4]
root = Path(sys.argv[4] if len(sys.argv) > 4 else "/e/fscratch/profound/naeimitabiei1/sliced-prof")
WINDOWS = {1: "~5K", 2: "~50K"}


def steps(db):
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    per = collections.defaultdict(list)
    for k in c.execute("select start, end, shortName, correlationId, gridX from "
                       "CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0"):
        per[k[3]].append(k)
    out = []
    for v in per.values():
        if len(v) < 500:
            continue
        nm = [names[k[2]] for k in v]
        m = [k[4] for k, n in zip(v, nm) if n.startswith("cooperative_topk")]
        if not m or m[0] != 16:
            continue
        dur = [(k[1] - k[0]) / 1e3 for k in v]
        moe = sum(d for d, n in zip(dur, nm) if n == "layer_kernel")
        fm = sum(d for d, n in zip(dur, nm) if "sparse" in n.lower() and "decode" in n.lower())
        span = (max(k[1] for k in v) - min(k[0] for k in v)) / 1e3
        out.append(dict(span=span, moe=moe, rest=span - moe, fmla=fm))
    return out


res = collections.defaultdict(dict)  # (win, hold) -> arm -> steps
for d in sorted(root.glob(f"{prefix}[{base}{arms}]-dec4-*")):
    arm, hold = d.name[len(prefix)], d.name.split("-")[-1]
    for db in sorted(d.glob("nsys.*.sqlite")):
        res[(int(db.name.split(".")[1]), hold)][arm] = steps(db)

F = ("span", "moe", "rest", "fmla")
pairs = []
print(f"{'win':5s} {'hold':8s} arm {'n':>7s} {'span base':>9s} " + " ".join(f"{f + ' x-b':>9s}" for f in F)
      + "   (ms; fmla us)")
for (win, hold), by in sorted(res.items()):
    b = by.get(base)
    if not b or len(b) < 15:
        continue
    mb = {f: st.median(s[f] for s in b) for f in F}
    for arm in arms:
        x = by.get(arm)
        if not x or len(x) < 15:
            continue
        d = {f: st.median(s[f] for s in x) - mb[f] for f in F}
        pairs.append(dict(win=win, hold=hold, arm=arm, n=len(x), base=mb, **d))
        print(f"{WINDOWS[win]:5s} {hold:8s}  {arm}  {len(b):3d}/{len(x):<3d} {mb['span'] / 1e3:9.2f} "
              + " ".join(f"{d[f] / (1 if f == 'fmla' else 1e3):+9.3f}" for f in F))
for arm in arms:
    for win in WINDOWS:
        sel = [p for p in pairs if p["arm"] == arm and p["win"] == win]
        if sel:
            sp = [p["span"] / 1e3 for p in sel]
            print(f"{arm} - {base} {WINDOWS[win]:5s}: span {st.mean(sp):+.3f} ms "
                  f"({min(sp):+.3f} to {max(sp):+.3f}, {len(sp)} holds)  "
                  f"moe {st.mean(p['moe'] for p in sel) / 1e3:+.3f}  "
                  f"rest {st.mean(p['rest'] for p in sel) / 1e3:+.3f}  "
                  f"fmla {st.mean(p['fmla'] for p in sel):+.1f} us")
(Path(__file__).resolve().parent / f"{prefix}_pairs.json").write_text(json.dumps(pairs, indent=1))
