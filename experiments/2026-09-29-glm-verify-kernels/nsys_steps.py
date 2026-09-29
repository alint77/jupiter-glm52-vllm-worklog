#!/usr/bin/env python3
"""Per-step host/GPU timeline from an nsys SQLite export (graphs traced whole).

Per rank: verify-graph executions (CUPTI_ACTIVITY_KIND_GRAPH_TRACE) delimit the
steps. For each step: period, graph time, other GPU work (drafter, sampling,
input prep), GPU idle, and the idle right before the graph split into
host-late (the cudaGraphLaunch call returned after the GPU had drained) or not.

    nsys_steps.py <export.sqlite>
"""

import collections
import sqlite3
import statistics
import sys


def union(iv):
    tot, cur = 0, None
    for s, e in sorted(iv):
        if cur is None or s > cur:
            tot += e - s
            cur = e
        elif e > cur:
            tot += e - cur
            cur = e
    return tot


def main():
    c = sqlite3.connect(sys.argv[1])
    names = dict(c.execute("select id, value from StringIds"))
    graphs = collections.defaultdict(list)
    for s, e, dev, corr in c.execute(
            "select start, end, deviceId, correlationId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        graphs[dev].append((s, e, corr))
    gpu = collections.defaultdict(list)
    for tbl in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for s, e, dev in c.execute(f"select start, end, deviceId from {tbl}"):
            gpu[dev].append((s, e))
    launch = {}
    api = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, e, corr, nid, tid in c.execute(
            "select start, end, correlationId, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        launch[corr] = (s, e, names[nid])
    print("rank steps  period  graph  other   idle  idle-before-graph  host-late  "
          "graphLaunch call (p50/p90/max, us)")
    for dev in sorted(graphs):
        gs = sorted(graphs[dev])
        ops = sorted(gpu[dev])
        rows = []
        for (s, e, corr), (s2, _, _) in zip(gs, gs[1:]):
            period = s2 - s
            other = [(max(a, s), min(b, s2)) for a, b in ops if b > e and a < s2]
            busy = (e - s) + union([(max(a, e), b) for a, b in other if b > e])
            prev_end = max((b for a, b in ops if b <= s2 and b > e), default=e)
            gap = s2 - prev_end
            l = launch.get([x for x in gs if x[0] == s2][0][2])
            late = max(0, min(gap, (l[1] - prev_end))) if l else 0
            rows.append((period, e - s, busy - (e - s), period - busy, gap, late,
                         (l[1] - l[0]) if l else 0))
        m = lambda k: statistics.fmean(r[k] for r in rows) / 1e6
        gl = sorted(r[6] / 1e3 for r in rows)
        print(f"{dev:4d} {len(rows):5d} {m(0):7.2f} {m(1):6.2f} {m(2):6.2f} {m(3):6.2f} "
              f"{m(4):12.3f} {m(5):14.3f}   {gl[len(gl)//2]:6.0f} / {gl[int(.9*len(gl))]:6.0f} / "
              f"{gl[-1]:6.0f}")
    top = collections.Counter()
    dur = collections.defaultdict(float)
    for s, e, n in launch.values():
        top[n] += 1
        dur[n] += e - s
    print("\nruntime API time per call type (all ranks, ms total / count):")
    for n in sorted(dur, key=lambda k: -dur[k])[:12]:
        print(f"  {dur[n] / 1e6:8.2f} ms  x{top[n]:6d}  {n}")


if __name__ == "__main__":
    main()
