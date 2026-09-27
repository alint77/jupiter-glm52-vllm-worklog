#!/usr/bin/env python3
"""Sum ncu --csv gpu__time_duration per kernel role, median over calls.

    ncu_sum.py marlin <csv>...   -> per file: align, w13, act, w2, sum, gemm+act+sum (us)
    ncu_sum.py sk <csv>...       -> per file: median sk_kernel duration (us)
"""

import csv
import json
import statistics
import sys


def rows(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("==")]
    lines = lines[next(i for i, ln in enumerate(lines) if ln.startswith('"ID"')):]
    for r in csv.DictReader(lines):
        if r.get("Metric Name") == "gpu__time_duration.sum":
            unit = r.get("Metric Unit", "ns")
            v = float(r["Metric Value"].replace(",", ""))
            yield r["Kernel Name"], v / 1000.0 if unit == "ns" else v


def marlin(path):
    calls, cur = [], None
    for name, us in rows(path):
        if "moe_align_block_size" in name:
            cur = {"align": us, "marlin": [], "act": 0.0, "sum": 0.0}
            calls.append(cur)
        elif "count_and_sort" in name:
            cur["align"] += us
        elif "Marlin<" in name:
            cur["marlin"].append(us)
        elif "act_and_mul" in name:
            cur["act"] += us
        elif "moe_sum" in name:
            cur["sum"] += us
    med = lambda f: statistics.median(f(c) for c in calls)  # noqa: E731
    return {"calls": len(calls), "align": med(lambda c: c["align"]), "w13": med(lambda c: c["marlin"][0]),
            "act": med(lambda c: c["act"]), "w2": med(lambda c: c["marlin"][1]), "sum": med(lambda c: c["sum"]),
            "total": med(lambda c: sum(c["marlin"]) + c["act"] + c["sum"])}


def sk(path):
    v = [us for name, us in rows(path) if "sk_kernel" in name]
    return {"calls": len(v), "us": statistics.median(v)}


if __name__ == "__main__":
    fn = marlin if sys.argv[1] == "marlin" else sk
    print(json.dumps({p: fn(p) for p in sys.argv[2:]}, indent=1))
