"""Stall reasons summed over SASS offset ranges.  sass_reg.py csv lo-hi ..."""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hdr, data = rows[1], rows[2:]
base = min(int(r[0], 16) for r in data if r[0].startswith("0x"))
sc = [i for i, h in enumerate(hdr) if h.startswith("stall_") and "Not Issued" not in h]
tot = sum(float(r[4] or 0) for r in data)
for rg in sys.argv[2:]:
    lo, hi = (int(x, 16) for x in rg.split("-"))
    sel = [r for r in data if lo <= int(r[0], 16) - base <= hi]
    s = sum(float(r[4] or 0) for r in sel)
    reasons = sorted(((sum(float(r[i] or 0) for r in sel), hdr[i][6:]) for i in sc), reverse=True)
    ops = {}
    for r in sel:
        op = r[1].split()[0] if not r[1].startswith("@") else r[1].split()[1]
        ops[op.split(".")[0]] = ops.get(op.split(".")[0], 0) + int(r[5] or 0)
    print(f"{rg}: {s / tot * 100:.1f}% of samples, {len(sel)} instr; "
          + " ".join(f"{n}:{v / s * 100:.0f}%" for v, n in reasons[:7]))
    print("   executed:", " ".join(f"{k}:{v}" for k, v in sorted(ops.items(), key=lambda x: -x[1])[:10]))
