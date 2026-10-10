"""breakdown2.py's decode step breakdown from an Nsight Systems report with
graph kernels traced per node (nsys_node.sh): no torch-profiler CUPTI cost.

A step runs from one verify-graph launch's first kernel to the next's (the
verify launch: a cudaGraphLaunch whose correlation id carries >= 500
kernels). Each instant goes to the highest-priority category running (verify
graph kernels by name, any other GPU work "outside the verify graph"), none
running is idle. One device (rank).

    nsys_bd.py <report.sqlite | report.nsys-rep> [--device N] [--json out]
"""
import argparse
import collections
import json
import sqlite3
import statistics as st
import subprocess
from pathlib import Path

from breakdown2 import CATS, NAMES, cat

NSYS = "/e/software/default/stages/2026/software/Nsight-Systems/2025.5.1-GCCcore-14.3.0/bin/nsys"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--device", type=int)
    ap.add_argument("--json")
    a = ap.parse_args()
    p = Path(a.report)
    if p.suffix != ".sqlite":
        q = p.with_suffix(".sqlite")
        if not q.exists():
            subprocess.run([NSYS, "export", "--type", "sqlite", "-o", str(q), str(p)],
                           check=True, capture_output=True)
        p = q
    c = sqlite3.connect(p)
    names = dict(c.execute("select id, value from StringIds"))
    dev = a.device if a.device is not None else min(
        d for (d,) in c.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL"))
    by_corr = collections.defaultdict(list)
    for s, e, corr, n in c.execute("select start, end, correlationId, demangledName from "
                                   "CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ?", (dev,)):
        by_corr[corr].append((s, e, names[n]))
    other = []
    for tbl in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        other += [(s, e, len(CATS)) for s, e in
                  c.execute(f"select start, end from {tbl} where deviceId = ?", (dev,))]
    verify = sorted((min(k[0] for k in ks), corr) for corr, ks in by_corr.items() if len(ks) >= 500)
    allk = sorted([(s, e, len(CATS), corr) for corr, ks in by_corr.items() for s, e, _ in ks]
                  + [(s, e, cc, None) for s, e, cc in other])
    vset = {corr for _, corr in verify}
    rows = []
    j = 0
    for (a0, corr), (b0, _) in zip(verify, verify[1:]):
        if b0 - a0 > 200e6:  # window gap, not a step
            continue
        ks = [(s, e, cat(n)) for s, e, n in by_corr[corr]]
        while j < len(allk) and allk[j][1] <= a0:
            j += 1
        k = j
        while k < len(allk) and allk[k][0] < b0:
            s, e, cc, kc = allk[k]
            if kc not in vset:
                ks.append((s, e, cc))
            k += 1
        pts = sorted({a0, b0} | {t for s, e, _ in ks if e > a0 and s < b0
                                 for t in (max(s, a0), min(e, b0))})
        acc = [0.0] * len(NAMES)
        ev = sorted(ks)
        for x, y in zip(pts, pts[1:]):
            m = (x + y) / 2
            act = [cc for s, e, cc in ev if s <= m < e]
            acc[min(act) if act else len(NAMES) - 1] += y - x
        rows.append(acc)
    mean = [st.mean(r[i] for r in rows) / 1e6 for i in range(len(NAMES))]
    total = sum(mean)
    print(f"device {dev}: {len(rows)} steps, mean period {total:.2f} ms "
          f"(p10 {sorted(sum(r) for r in rows)[len(rows) // 10] / 1e6:.2f}, "
          f"p90 {sorted(sum(r) for r in rows)[len(rows) * 9 // 10] / 1e6:.2f})")
    for n, v in sorted(zip(NAMES, mean), key=lambda x: -x[1]):
        print(f"  {v:6.2f} ms  {100 * v / total:5.1f}%  {n}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"steps": len(rows), "period_ms": total, "ms": dict(zip(NAMES, mean)),
                       "per_step_ms": [[v / 1e6 for v in r] for r in rows]}, f)


if __name__ == "__main__":
    main()
