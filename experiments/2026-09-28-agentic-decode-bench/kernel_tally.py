#!/usr/bin/env python3
"""Independent cross-check for analyze.py: rank-0 GPU kernels per decode step.

Steps are counted from the scheduler's execute_* annotations; kernel time is
summed per kernel name (no phase or category logic), plus GPU busy (union of
kernel spans) and the step period, so a category table can be checked against
the raw kernels.

    kernel_tally.py <window dir>... [--top 30]
"""

import argparse
import collections
import gzip
import json
from pathlib import Path


def union(spans: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for a, b in sorted(spans):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    per_name = collections.defaultdict(float)
    steps = busy = span = 0.0
    for d in args.dirs:
        f = sorted(d.glob("*rank0*.pt.trace.json.gz"))[0]
        events = [e for e in json.load(gzip.open(f, "rt"))["traceEvents"] if e.get("ph") == "X"]
        marks = sorted(e["ts"] for e in events if e.get("cat") == "user_annotation"
                       and e["name"].startswith("execute_"))
        kernels = [e for e in events if e.get("cat") == "kernel"
                   and marks[0] <= e["ts"] < marks[-1]]
        n = len(marks) - 1
        steps += n
        span += (marks[-1] - marks[0]) / 1000
        busy += union([(e["ts"], e["ts"] + e["dur"]) for e in kernels]) / 1000
        for e in kernels:
            per_name[e["name"][:110]] += e["dur"] / 1000
    print(f"{steps:.0f} steps over {len(args.dirs)} window(s): "
          f"period {span / steps:.2f} ms, GPU busy {busy / steps:.2f} ms, "
          f"idle {(span - busy) / steps:.2f} ms per step")
    total = sum(per_name.values())
    print(f"kernel time {total / steps:.2f} ms per step (sum, overlaps counted twice)")
    for name, t in sorted(per_name.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"  {t / steps:7.3f} ms  {name}")


if __name__ == "__main__":
    main()
