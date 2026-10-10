"""Median us per variant x cell from kdev bench lines (possibly truncated).

    abl_sum.py <file>...
"""
import collections
import re
import statistics as st
import sys

d = collections.defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f):
        m = re.search(r'"v": "([^"]*)", "m": (\d+), "hot": (\d+), "cold": (\d+).*?"us": ([\d.]+)', l)
        if m:
            v = re.sub(r"^td_", "", m[1]).replace("TD_COMPUTE_ONLY", "CO").replace("TD_ABL_", "") or "full"
            d[(v, int(m[2]), int(m[3]), int(m[4]))].append(float(m[5]))
cells = sorted({k[1:] for k in d})
vs = list(dict.fromkeys(k[0] for k in d))
print("variant".ljust(34) + "".join(f"M{m} {h}/{c}".rjust(12) for m, h, c in cells))
for v in vs:
    print(v.ljust(34) + "".join(f"{st.median(d[(v,) + c]):12.1f}" if (v,) + c in d else " " * 12 for c in cells))
