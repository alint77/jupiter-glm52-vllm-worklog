#!/usr/bin/env python3
"""Is reduce-scatter waiting explained by per-layer MoE imbalance across ranks?

Splits each step's target graph into per-layer segments at the reduce-scatter
that closes every MoE layer. For layer k on rank r, "arrival" is when rank r
launches reduce-scatter k (relative to when that reduce-scatter completes,
which is synchronous across ranks). Waiting is the latest arrival minus this
rank's arrival. It is compared with the cross-rank spread of each rank's own
Marlin time (hot and cold, per tier) in the same layer.

    imbalance.py <trace-dir>
"""

import collections
import statistics
import sys
from pathlib import Path

from analyze import RANK_RE, category, load, steps, union


def layer_rows(rows):
    out = []
    for row in rows:
        ops = sorted(row["phases"]["target"], key=lambda e: e["t"])
        rs = [o for o in ops if "ReduceScatter" in o["name"]]
        left = ops[0]["t"]
        layers = []
        for r in rs:
            seg = [o for o in ops if left <= o["t"] < r["t"] and "ReduceScatter" not in o["name"]]
            hot = [o for o in seg if category(o).startswith("MoE hot")]
            cold = [o for o in seg if category(o).startswith("MoE cold")]
            layers.append({"arrive": r["t"], "done": r["end"],
                           "hot": union(hot), "cold": union(cold), "marlin": union(hot + cold)})
            left = r["end"]
        out.append(layers)
    return out


def main():
    trace_dir = Path(sys.argv[1])
    per_rank = {}
    for path in sorted(trace_dir.glob("*.pt.trace.json.gz")):
        per_rank[int(RANK_RE.search(path.name).group(1))] = layer_rows(steps(load(path)))
    n_steps = min(len(v) for v in per_rank.values())
    wait = collections.defaultdict(float)
    spread = collections.defaultdict(float)
    slowest = collections.Counter()
    cold_spread = hot_spread = 0.0
    last_arrivals = 0
    for s in range(n_steps):
        n_layers = min(len(per_rank[r][s]) for r in per_rank)
        for k in range(n_layers):
            rows = {r: per_rank[r][s][k] for r in per_rank}
            # Completion is synchronous; measure arrival relative to it.
            lateness = {r: row["done"] - row["arrive"] for r, row in rows.items()}
            latest = min(lateness.values())  # the rank that waited least arrived last
            for r in rows:
                wait[r] += lateness[r] - latest
            marlin = {r: row["marlin"] for r, row in rows.items()}
            mean_marlin = statistics.fmean(marlin.values())
            for r in rows:
                spread[r] += max(marlin.values()) - marlin[r]
            slowest[min(lateness, key=lateness.get)] += 1
            cold_spread += max(row["cold"] for row in rows.values()) - statistics.fmean(row["cold"] for row in rows.values())
            hot_spread += max(row["hot"] for row in rows.values()) - statistics.fmean(row["hot"] for row in rows.values())
            last_arrivals += 1
    print(f"{trace_dir.name}: {n_steps} steps")
    print("per rank, ms/step: waiting in reduce-scatter vs (slowest rank's Marlin - own Marlin)")
    for r in sorted(per_rank):
        print(f"  rank {r}: waiting {wait[r] / n_steps:5.2f}   Marlin deficit {spread[r] / n_steps:5.2f}")
    print(f"mean waiting {statistics.fmean(wait.values()) / n_steps:.2f}, "
          f"mean Marlin deficit {statistics.fmean(spread.values()) / n_steps:.2f} ms/step")
    print(f"per-layer spread (max - mean over ranks), summed per step: cold {cold_spread / n_steps:.2f} ms, "
          f"hot {hot_spread / n_steps:.2f} ms")
    total = sum(slowest.values())
    print("last to arrive, share of layers:", {r: f"{c / total:.0%}" for r, c in sorted(slowest.items())})


if __name__ == "__main__":
    main()
