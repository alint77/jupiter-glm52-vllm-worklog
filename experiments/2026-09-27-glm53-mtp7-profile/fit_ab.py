#!/usr/bin/env python3
"""Least-squares A/B over ab-*.json runs: step_ms ~ arm + tokens/step + node + context.

    fit_ab.py BASE_ARM TAG... (e.g. fit_ab.py off rA rB)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
base, tags = sys.argv[1], sys.argv[2:]
rows = []
for tag in tags:
    for f in sorted(HERE.glob(f"ab-{tag}-*.json")):
        arm = f.stem.split("-", 2)[2]
        for ctx, v in json.loads(f.read_text()).items():
            if ctx.startswith("ttft"):
                continue
            for r in v["runs"]:
                rows.append((tag, arm, ctx, r["step_ms"], r["tokens_per_step"]))
arms = sorted({r[1] for r in rows} - {base})
cols = ["intercept", "tokens/step", *[f"{a} vs {base}" for a in arms],
        *[f"node {t}" for t in tags[1:]], "96k vs short"]
X = np.array([[1.0, r[4], *[float(r[1] == a) for a in arms],
               *[float(r[0] == t) for t in tags[1:]], float(r[2] == "96k")] for r in rows])
y = np.array([r[3] for r in rows])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ coef
cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(y) - X.shape[1])
print(f"{len(rows)} runs; residual sd {res.std(ddof=X.shape[1]):.2f} ms")
for name, c, s in zip(cols, coef, np.sqrt(np.diag(cov))):
    print(f"  {name:22s} {c:7.2f} +- {s:.2f}")
