#!/usr/bin/env python3
"""Does step time follow each step's accepted-token count, or only the text?

Reads the (perf_counter, accepted) lines the env-gated scheduler hook writes
(step-trace.patch, VLLM_STEP_TRACE_FILE) at c=1. A step's wall time is the gap
to the next line; gaps over 0.2 s are request boundaries and dropped. The time
is regressed on the accepted count of the step it follows, the one before and
the one after. A mechanical per-token cost shows up on lag 0 or 1; a content
effect (text that accepts well also routes to more cold experts) shows up
smeared across all lags, and in the slow drift of a rolling window.

    steptrace.py steptrace-<job>.csv
"""

import sys

import numpy as np

rows = np.loadtxt(sys.argv[1], delimiter=",")
t, acc = rows[:, 0], rows[:, 1]
dt = np.diff(t) * 1000
ok = dt < 200
idx = np.arange(1, len(dt) - 1)
idx = idx[ok[idx] & ok[idx - 1] & ok[idx + 1]]
y = dt[idx]
X = np.column_stack([np.ones(len(idx)), acc[idx], acc[idx - 1], acc[idx + 1]])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ coef
cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(y) - X.shape[1])
print(f"{len(y)} steps, median {np.median(y):.2f} ms, residual sd {res.std():.2f}")
for name, c, s in zip(["intercept", "accepted, this step", "accepted, step before",
                       "accepted, step after"], coef, np.sqrt(np.diag(cov))):
    print(f"  {name:22s} {c:7.3f} +- {s:.3f} ms per token")
print("median step by accepted count (this step):")
for a in range(int(acc.max()) + 1):
    sel = acc[idx] == a
    if sel.sum() >= 10:
        print(f"  {a}: {np.median(y[sel]):6.2f} ms  (n={sel.sum()})")
# the slow part: 32-step rolling means of both
k = 32
if len(y) > 4 * k:
    ry = np.convolve(y, np.ones(k) / k, "valid")
    ra = np.convolve(acc[idx], np.ones(k) / k, "valid")
    slope = np.polyfit(ra, ry, 1)[0]
    print(f"32-step rolling means: {slope:.3f} ms per accepted token (corr "
          f"{np.corrcoef(ra, ry)[0, 1]:.2f})")
