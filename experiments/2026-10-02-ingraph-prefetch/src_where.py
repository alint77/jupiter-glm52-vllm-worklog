"""Per launch: share of stall samples by instruction kind, and the top
instructions with their position in the SASS listing (prologue waits come
first, the main loop's later)."""
import csv
import re
import sys
from collections import defaultdict


def num(x):
    try:
        return float(x.replace(",", "") or 0)
    except ValueError:
        return None


for path in sys.argv[1:]:
    rows = list(csv.reader(open(path)))
    name = rows[0][1] if len(rows[0]) > 1 else path
    hd = rows[1]
    si, sa = hd.index("Source"), hd.index("Warp Stall Sampling (All Samples)")
    rc = [i for i, c in enumerate(hd) if c.startswith("stall_") and "Not Issued" not in c]
    body = [r for r in rows[2:] if len(r) > sa and num(r[sa]) is not None]
    body = body[: len(body) // 2] if len(body) % 2 == 0 and body[: len(body) // 2] == body[len(body) // 2:] else body
    tot = sum(num(r[sa]) for r in body) or 1
    print(f"== {name[:60]}  ({len(body)} SASS lines)")
    kinds = defaultdict(float)
    for r in body:
        m = re.match(r"(@!?U?P\w+\s+)?([A-Z0-9_]+)", r[si].strip())
        kinds[m.group(2) if m else "?"] += num(r[sa])
    print("   " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:8]))
    for i, r in sorted(enumerate(body), key=lambda ir: -num(ir[1][sa]))[:10]:
        rs = sorted(((hd[c][6:], num(r[c]) or 0) for c in rc if c < len(r)), key=lambda kv: -kv[1])[:2]
        print(f"   {100 * num(r[sa]) / tot:5.1f}%  line {i:5d}  {r[si].strip()[:52]:52s} " + ", ".join(f"{a} {100 * b / max(num(r[sa]), 1):.0f}%" for a, b in rs))
