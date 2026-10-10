"""SASS window with samples and top stall reasons per instruction.
    sass_win.py <ncu sass csv> <lo hex offset> <hi hex offset>"""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[1]
data = rows[2:]
base = min(int(r[0], 16) for r in data if r[0].startswith("0x"))
lo, hi = int(sys.argv[2], 16), int(sys.argv[3], 16)
sc = [i for i, h in enumerate(hdr) if h.startswith("stall_") and "Not Issued" not in h]
tot = sum(float(r[4] or 0) for r in data)
for r in data:
    off = int(r[0], 16) - base
    if lo <= off <= hi:
        s = float(r[4] or 0)
        top = sorted(((float(r[i] or 0), hdr[i][6:]) for i in sc), reverse=True)[:3]
        print(f"{off:05x} {s / tot * 100:5.2f}% exe {r[5]:>6}  {r[1][:62]:62s} "
              + " ".join(f"{n}:{v:.0f}" for v, n in top if v > 0))
