#!/usr/bin/env python3
"""All-reduce cost without the cross-rank wait: per step and per all-reduce
site (ordinal within the verify graph), the fastest rank's kernel duration is
the transfer + fused-work time; the spread above it is waiting for the slowest
rank. Steps are aligned across ranks by index within the window.

    ar_transfer.py <window dir> [<name substring>...]
"""
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402

win = Path(sys.argv[1])
names = sys.argv[2:] or ["cross_device_reduce", "allreduce_fusion"]
per_rank = {}
for f in sorted(win.glob("*rank*.pt.trace.json.gz")):
    r = int(f.name.split("_rank")[1].split(".")[0])
    rows = A.steps(A.load(f))
    per_rank[r] = [[o["end"] - o["t"] for o in sorted(row["phases"]["target"], key=lambda o: o["t"])
                    if any(n in o["name"] for n in names)] for row in rows]
n = min(len(v) for v in per_rank.values())
mins, means, sites = [], [], collections.Counter()
for s in range(n):
    lists = [per_rank[r][s] for r in per_rank]
    k = min(len(l) for l in lists)
    sites[k] += 1
    for i in range(k):
        vals = [l[i] for l in lists]
        mins.append(min(vals))
        means.append(statistics.fmean(vals))
steps = n
print(f"{win.name}: {steps} steps, {statistics.fmean(sites.elements()):.0f} sites/step")
print(f"  per site: fastest rank {statistics.fmean(mins)*1000:.1f} us, rank mean {statistics.fmean(means)*1000:.1f} us")
print(f"  per step: fastest-rank sum {sum(mins)/steps:.3f} ms, mean {sum(means)/steps:.3f} ms, wait {(sum(means)-sum(mins))/steps:.3f} ms")
