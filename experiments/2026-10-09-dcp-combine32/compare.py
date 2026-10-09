"""Before / after per hold (same node), per (ctx, n): step ms and total tok/s
from the sweep arms (m32-sweep/cmb-<arm>-<job>-<i>/sweep.jsonl).

    compare.py
"""
import collections
import json
import statistics as st
from pathlib import Path

ROOT = Path("/e/fscratch/profound/naeimitabiei1/m32-sweep")
cell = collections.defaultdict(lambda: collections.defaultdict(list))
for d in sorted(ROOT.glob("cmb-*")):
    f = d / "sweep.jsonl"
    if not f.exists():
        continue
    _, arm, job, _ = d.name.split("-")
    for l in f.read_text().splitlines():
        r = json.loads(l)
        cell[(r["ctx_target"], r["n"])][(job, arm)].append(r)
print(f"{'ctx':>6} {'n':>2} {"median step before":>18} {'after':>8} {'delta':>7}  {'tok/s before':>12} {'after':>7}  pairs")
for (ctx, n), by in sorted(cell.items()):
    jobs = sorted({j for j, _ in by if (j, "before") in by and (j, "after") in by})
    if not jobs:
        continue
    sb = st.mean(st.median(r["step_ms"] for r in by[(j, "before")]) for j in jobs)
    sa = st.mean(st.median(r["step_ms"] for r in by[(j, "after")]) for j in jobs)
    tb = st.mean(st.mean(r["agg_tps"] for r in by[(j, "before")]) for j in jobs)
    ta = st.mean(st.mean(r["agg_tps"] for r in by[(j, "after")]) for j in jobs)
    print(f"{ctx:6d} {n:2d} {sb:18.2f} {sa:8.2f} {sa - sb:+7.2f}  {tb:12.1f} {ta:7.1f}  {len(jobs)}")
