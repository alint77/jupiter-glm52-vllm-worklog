#!/usr/bin/env python3
"""Pool every GSM8K run per arm (gsm-<tag>-<arm>.json) over the questions all runs
share, and bootstrap the new - old accuracy over questions.   gsm_pool.py TAG..."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
runs = {"new": [], "old": []}
for tag in sys.argv[1:]:
    for arm in runs:
        d = json.loads((HERE / f"gsm-{tag}-{arm}.json").read_text())
        runs[arm].append(np.asarray(d["correct"], dtype=float))
n = min(len(r) for rs in runs.values() for r in rs)
m = {arm: np.stack([r[:n] for r in rs]) for arm, rs in runs.items()}
for arm, a in m.items():
    print(f"{arm}: {len(a)} runs on {n} shared questions, " +
          ", ".join(f"{x:.4f}" for x in a.mean(1)) + f"; pooled {a.mean():.4f}")
rng = np.random.default_rng(0)
per_q = m["new"].mean(0) - m["old"].mean(0)
boot = [per_q[rng.integers(0, n, n)].mean() for _ in range(20000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"new - old: {per_q.mean() * 100:+.2f} pts, 95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}]")
old = m["old"]
flips = [(old[i] != old[j]).sum() for i in range(len(old)) for j in range(i + 1, len(old))]
print(f"old against itself: {np.mean(flips):.0f} answers differ per pair of runs")
