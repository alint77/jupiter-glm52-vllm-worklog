"""Top SASS instructions by warp stall samples from an ncu source csv
(ncu_src.sh), with each one's main stall reasons:  src_top.py <csv> [n]"""
import csv
import sys

lines = open(sys.argv[1]).read().splitlines()[1:]
rows = list(csv.DictReader(lines))
key = "Warp Stall Sampling (All Samples)"
reasons = [k for k in rows[0] if k.startswith("stall_") and "Not Issued" not in k]
f = lambda v: float(v) if v not in ("", "-") else 0.0
tot = sum(f(r[key]) for r in rows)
by = {k: sum(f(r[k]) for r in rows) for k in reasons}
print(f"total samples {tot:.0f}: " + ", ".join(
    f"{k[6:]} {100 * v / tot:.0f}%" for k, v in sorted(by.items(), key=lambda x: -x[1]) if v / tot > 0.01))
for r in sorted(rows, key=lambda r: -f(r[key]))[: int(sys.argv[2]) if len(sys.argv) > 2 else 30]:
    s = f(r[key])
    why = ", ".join(f"{k[6:]} {100 * f(r[k]) / s:.0f}" for k in sorted(reasons, key=lambda k: -f(r[k]))[:3] if f(r[k]) > 0)
    print(f"{100 * s / tot:5.1f}%  {r['Address'][-5:]}  {r['Source'].strip()[:60]:60s}  {why}")
