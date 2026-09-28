#!/usr/bin/env python3
"""Pool analyze.py over a model's profiler windows (step-weighted), folding
Inductor matmul templates (triton_tem_fused_mm*) from norm/rope/elementwise into
dense GEMM and the MoE one-kernel chain into one MoE row (PDL overlaps its
kernels, so only the chain's total is meaningful).   pool_windows.py <dir>..."""
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402

MOE = ("MoE one-kernel w13", "MoE one-kernel w2", "MoE one-kernel route/act/finalize",
       "MoE routing/align/sum/act", "MoE hot Marlin (HBM)", "MoE cold Marlin (Grace)")
tot = collections.defaultdict(float)
n = 0
for d in map(Path, sys.argv[1:]):
    ranks = {}
    for path in sorted(d.glob("*.pt.trace.json.gz")):
        ranks[path.name] = A.steps(A.load(path))
    for rows in ranks.values():
        for row in rows:
            n += 1
            tot["period"] += row["period"]
            for name, ops in row["phases"].items():
                tot[f"phase {name}"] += A.union(ops)
            tem = sum(o["end"] - o["t"] for o in row["phases"]["target"]
                      if o["name"].startswith("triton_tem_fused_mm"))
            for name, v in A.partition(row["phases"]["target"]).items():
                tot["MoE" if name in MOE else name] += v
            tot["norm/rope/elementwise"] -= tem
            tot["dense GEMM"] += tem
busy = sum(v for k, v in tot.items() if k.startswith("phase "))
print(f"{n} rank-steps; period {tot['period'] / n:.2f} ms")
for k in ("phase draft", "phase logits", "phase host-side"):
    print(f"  {k[6:]:28s} {tot[k] / n:6.2f}")
print(f"  {'GPU idle':28s} {(tot['period'] - busy) / n:6.2f}")
print("  target:")
for k, v in sorted(((k, v) for k, v in tot.items() if not k.startswith("phase ")
                    and k != "period"), key=lambda kv: -kv[1]):
    if v / n > 0.005:
        print(f"    {k:26s} {v / n:6.2f}")
