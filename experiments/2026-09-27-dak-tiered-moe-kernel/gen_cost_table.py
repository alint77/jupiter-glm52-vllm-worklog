#!/usr/bin/env python3
"""tgrid.json -> the C cost table embedded in tiered_decode.cu.

T[h][c], h = 0..24 hot and c = 0..6 cold experts on a GPU, in us: the measured
graph-replay layer time, linearly filled at the unmeasured hot counts, rounded
and made non-decreasing in both h and c (the reversal must never find that
removing an expert makes a GPU slower).
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / "tgrid.json").read_text())
grid = {(r["hot"], r["cold"]): r["us"] for r in rows}
grid[(0, 0)] = 0.0
hs = sorted({h for h, _ in grid})
T = np.zeros((25, 7))
for c in range(7):
    T[:, c] = np.interp(np.arange(25), hs, [grid[(h, c)] for h in hs])
T = np.maximum.accumulate(np.maximum.accumulate(T, axis=0), axis=1)
T = np.rint(T).astype(int)
print("constexpr unsigned short COST_US[COST_HOT][COST_COLD] = {")
for h in range(25):
    print("    {" + ", ".join(str(v) for v in T[h]) + "},")
print("};")
