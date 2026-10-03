"""Group ncu per-SASS stall samples by instruction kind (argv: raw csv, then
one --page source csv per launch)."""
import csv
import re
import sys
from collections import defaultdict

raw = list(csv.reader(open(sys.argv[1])))
h = raw[0]
for n, (src, lab) in enumerate(zip(sys.argv[2:], ("mode0 full", "mode2 compute-only"))):
    d = raw[2 + n]
    t = float(d[h.index("gpu__time_duration.sum")].replace(",", ""))
    ten = d[h.index("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")]
    rows = list(csv.reader(open(src)))
    hi = next(i for i, r in enumerate(rows) if "Source" in r)
    hd = rows[hi]
    si = hd.index("Source")
    samp = next(i for i, c in enumerate(hd) if c.startswith("Warp Stall Sampling (All"))
    reason_cols = [i for i, c in enumerate(hd) if c.startswith("stall_") and "Not Issued" not in c]
    ins = []
    groups = defaultdict(float)
    reasons = defaultdict(lambda: defaultdict(float))
    total = 0.0
    for r in rows[hi + 1:]:
        if len(r) <= samp:
            continue
        try:
            v = float(r[samp].replace(",", "") or 0)
        except ValueError:
            continue
        op = r[si].strip()
        m = re.match(r"(@!?U?P\w+\s+)?([A-Z0-9_.]+)", op)
        k = m.group(2) if m else op
        k = k.split(".")[0] if not k.startswith("SYNCS") else k[:22]
        groups[k] += v
        total += v
        ins.append((v, r[0], op))
        for i in reason_cols:
            if i >= len(r):
                continue
            try:
                reasons[k][hd[i]] += float(r[i].replace(",", "") or 0)
            except ValueError:
                pass
    print(f"{lab}: {t / 1e3:.0f} us, tensor {ten}%, {total:.0f} samples")
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1])[:12]:
        top = sorted(reasons[k].items(), key=lambda kv: -kv[1])[:3]
        print(f"   {k:24s} {100 * v / total:5.1f}%   " + ", ".join(f"{a[6:]} {100 * b / max(v, 1):.0f}%" for a, b in top))
    print("   top instructions:")
    for v, addr, op in sorted(ins, reverse=True)[:14]:
        print(f"     {100 * v / total:5.1f}%  {addr}  {op[:70]}")
