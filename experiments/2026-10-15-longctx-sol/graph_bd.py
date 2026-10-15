"""Decode verify-graph breakdown that works in windows where prefill chunks sit
between decode steps (nsys_bd.py drops those steps as window gaps): each
verify graph (a launch with >= 500 kernels) is measured over its own span,
first kernel start to last kernel end, and every instant of the span goes to
the highest-priority category running on the device (nsys_bd.py's rule and
categories), none running is idle. Also prints the per-kernel-name mean time
inside the span for the largest names.

    graph_bd.py report.sqlite [--device N] [--json out] [--top 25]
"""
import argparse
import collections
import json
import sqlite3
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-09-tp-sliced-experts"))
from breakdown2 import CATS, NAMES, cat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--json")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    c = sqlite3.connect(a.report)
    names = dict(c.execute("select id, value from StringIds"))
    by_corr = collections.defaultdict(list)
    allk = []
    for s, e, corr, n in c.execute("select start, end, correlationId, demangledName from "
                                   "CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ?", (a.device,)):
        by_corr[corr].append((s, e, names[n]))
        allk.append((s, e, names[n]))
    allk.sort()
    graphs = sorted((min(k[0] for k in ks), max(k[1] for k in ks), corr)
                    for corr, ks in by_corr.items() if len(ks) >= 500)
    rows, per_name = [], collections.defaultdict(float)
    j = 0
    for g0, g1, corr in graphs:
        while j < len(allk) and allk[j][1] <= g0:
            j += 1
        ks = []
        k = j
        while k < len(allk) and allk[k][0] < g1:
            s, e, n = allk[k]
            if e > g0:
                ks.append((max(s, g0), min(e, g1), cat(n), n))
            k += 1
        for s, e, _, n in ks:
            per_name[n] += e - s
        pts = sorted({g0, g1} | {t for s, e, _, _ in ks for t in (s, e)})
        acc = [0.0] * len(NAMES)
        for x, y in zip(pts, pts[1:]):
            m = (x + y) / 2
            act = [cc for s, e, cc, _ in ks if s <= m < e]
            acc[min(act) if act else len(NAMES) - 1] += y - x
        rows.append(acc)
    n = len(rows)
    mean = [st.mean(r[i] for r in rows) / 1e6 for i in range(len(NAMES))]
    spans = sorted(sum(r) / 1e6 for r in rows)
    print(f"device {a.device}: {n} verify graphs, mean span {sum(mean):.2f} ms "
          f"(p10 {spans[n // 10]:.2f}, p90 {spans[n * 9 // 10]:.2f})")
    for nm, v in sorted(zip(NAMES, mean), key=lambda x: -x[1]):
        if v > 0.005:
            print(f"  {v:6.2f} ms  {100 * v / sum(mean):5.1f}%  {nm}")
    print("  top kernels (summed over streams, ms per graph):")
    for nm, v in sorted(per_name.items(), key=lambda x: -x[1])[:a.top]:
        print(f"    {v / n / 1e6:6.3f}  {nm[:110]}")
    if a.json:
        Path(a.json).write_text(json.dumps({
            "graphs": n, "span_ms": sum(mean), "ms": dict(zip(NAMES, mean)),
            "kernels_ms": {k: v / n / 1e6 for k, v in per_name.items()}}))


if __name__ == "__main__":
    main()
