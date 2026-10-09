"""Summarise ab-<tag>.jsonl: median us and roofline share per variant x cell.

    ab_sum.py logs/ab-<tag>.jsonl
"""
import collections
import json
import statistics as st
import sys

d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l)
    src, _, defs = r["v"].partition(":")
    defs = " ".join(x for x in defs.split() if not x.startswith(("TD_INTER", "TD_CHUNK", "TD_STAGES0", "TD_STAGES1")) or src != "td_v0")
    name = src + (f"[{defs}]" if defs else "") + ("+side" if r["shared"] == "side" else "")
    d[(name, r["hot"], r["cold"])].append(r)
cells = sorted({(k[1], k[2]) for k in d})
vs = sorted({k[0] for k in d})
w = max(len(v) for v in vs) + 2
print("variant".ljust(w) + "".join(f"{h}/{c}".rjust(9) + "  roof" for h, c in cells))
for v in vs:
    line = v.ljust(w)
    for h, c in cells:
        rs = d.get((v, h, c), [])
        line += (f"{st.median(r['us'] for r in rs):9.1f} {st.median(r.get('roof_share', 0) for r in rs):5.2f}"
                 if rs else " " * 15)
    print(line)
