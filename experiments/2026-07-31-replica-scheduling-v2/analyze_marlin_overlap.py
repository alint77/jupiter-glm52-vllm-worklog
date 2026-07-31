#!/usr/bin/env python3
"""Decide why v1's routed-Marlin cumulative residency rose 13.4%.

Two candidate causes for the +3.972 ms:

  added work   the greedy converted hot-primary experts into Grace secondaries,
               so the cold kernels read more weights;
  contention   balancing made the hot and cold tiers similar in length, so they
               overlap more, and concurrent kernels stretch each other. The sum
               of durations then rises while the union does not.

They are distinguishable per stream: added work lengthens the cold stream only
and raises the union of Marlin residency; contention lengthens both streams and
raises measured hot/cold overlap while the union stays flat or falls.

Filtering matches ``analyze_runtime_trace.py`` in the v1 experiment: only
steady ``_generation_4(16)`` decode graphs, selected by launch correlation.
"""

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

STEP_MARKER = "_generation_4(16)"
RANK_RE = re.compile(r"_rank(\d+)\.")
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
RUNTIME_CATEGORIES = {"cuda_runtime", "cuda_driver"}
MARLIN = "marlin_moe_wna16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", nargs=2, action="append", metavar=("NAME", "DIR"), required=True
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [e for e in json.load(handle)["traceEvents"] if e.get("ph") == "X"]


def union_us(spans: list[tuple[float, float]]) -> float:
    if not spans:
        return 0.0
    spans = sorted(spans)
    total = 0.0
    start, end = spans[0]
    for lo, hi in spans[1:]:
        if lo > end:
            total += end - start
            start, end = lo, hi
        else:
            end = max(end, hi)
    return total + end - start


def overlap_us(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Time during which at least one span from each list is live."""
    events = [(lo, 0, 1) for lo, _ in a] + [(hi, 0, -1) for _, hi in a]
    events += [(lo, 1, 1) for lo, _ in b] + [(hi, 1, -1) for _, hi in b]
    events.sort()
    depth = [0, 0]
    total = 0.0
    last = 0.0
    for time, which, delta in events:
        if depth[0] > 0 and depth[1] > 0:
            total += time - last
        depth[which] += delta
        last = time
    return total


def rank_steps(path: Path) -> list[dict]:
    events = load(path)
    by_correlation: dict[int, list[dict]] = collections.defaultdict(list)
    for event in events:
        correlation = event.get("args", {}).get("correlation")
        if event.get("cat") in GPU_CATEGORIES and correlation is not None:
            by_correlation[correlation].append(event)
    runtime = [e for e in events if e.get("cat") in RUNTIME_CATEGORIES]
    annotations = sorted(
        (
            e
            for e in events
            if e.get("cat") == "user_annotation" and STEP_MARKER in e["name"]
        ),
        key=lambda e: e["ts"],
    )

    steps = []
    for annotation in annotations:
        end = annotation["ts"] + annotation["dur"]
        launches = [
            e
            for e in runtime
            if annotation["ts"] <= e["ts"] < end and "GraphLaunch" in e["name"]
        ]
        if len(launches) != 1:
            raise ValueError(f"{path.name}: expected one decode graph")
        ops = by_correlation[launches[0]["args"]["correlation"]]
        if not ops:
            raise ValueError(f"{path.name}: decode graph has no device operations")

        marlin_by_stream: dict[int, list[tuple[float, float]]] = (
            collections.defaultdict(list)
        )
        for event in ops:
            if MARLIN in event["name"]:
                stream = int(event.get("args", {}).get("stream", -1))
                marlin_by_stream[stream].append(
                    (float(event["ts"]), float(event["ts"] + event["dur"]))
                )
        # The hot tier runs on the main stream and issues the majority of the
        # launches; the cold tier runs on the aux stream.
        order = sorted(marlin_by_stream, key=lambda s: -len(marlin_by_stream[s]))
        rows = {}
        for label, stream in zip(("hot", "cold"), order):
            spans = marlin_by_stream[stream]
            rows[label] = {
                "launches": len(spans),
                "cumulative_us": sum(hi - lo for lo, hi in spans),
                "union_us": union_us(spans),
            }
        all_marlin = [s for spans in marlin_by_stream.values() for s in spans]
        steps.append(
            {
                "marlin_launches": len(all_marlin),
                "marlin_streams": len(marlin_by_stream),
                "gpu_busy_us": union_us(
                    [(float(e["ts"]), float(e["ts"] + e["dur"])) for e in ops]
                ),
                "marlin_union_us": union_us(all_marlin),
                "marlin_cumulative_us": sum(hi - lo for lo, hi in all_marlin),
                "hot_cold_overlap_us": (
                    overlap_us(
                        marlin_by_stream[order[0]], marlin_by_stream[order[1]]
                    )
                    if len(order) >= 2
                    else 0.0
                ),
                "streams": rows,
            }
        )
    return steps


def summarize(directory: Path) -> dict:
    flat: list[dict] = []
    for path in sorted(Path(directory).glob("*.pt.trace.json.gz")):
        flat.extend(rank_steps(path))
    if not flat:
        raise ValueError(f"No steady decode graphs in {directory}")

    def mean(key: str) -> float:
        return sum(step[key] for step in flat) / len(flat) / 1000.0

    result = {
        "rank_steps": len(flat),
        "gpu_busy_ms": mean("gpu_busy_us"),
        "marlin_cumulative_ms": mean("marlin_cumulative_us"),
        "marlin_union_ms": mean("marlin_union_us"),
        "hot_cold_overlap_ms": mean("hot_cold_overlap_us"),
        "marlin_launches": sum(s["marlin_launches"] for s in flat) / len(flat),
        "marlin_streams": sum(s["marlin_streams"] for s in flat) / len(flat),
        # The decisive statistic: mean duration of every routed Marlin launch.
        # Added cold work would lift only the cold tier; a slower node lifts
        # every launch uniformly.
        "marlin_mean_us": sum(s["marlin_cumulative_us"] for s in flat)
        / sum(s["marlin_launches"] for s in flat),
    }
    for label in ("hot", "cold"):
        rows = [step["streams"][label] for step in flat if label in step["streams"]]
        if not rows:
            continue
        result[f"{label}_launches"] = sum(r["launches"] for r in rows) / len(rows)
        result[f"{label}_cumulative_ms"] = (
            sum(r["cumulative_us"] for r in rows) / len(rows) / 1000.0
        )
        result[f"{label}_union_ms"] = (
            sum(r["union_us"] for r in rows) / len(rows) / 1000.0
        )
        result[f"{label}_mean_us"] = sum(r["cumulative_us"] for r in rows) / sum(
            r["launches"] for r in rows
        )
    return result


def main() -> None:
    args = parse_args()
    results = {name: summarize(Path(directory)) for name, directory in args.arm}
    names = list(results)
    keys = list(results[names[0]])
    width = max(len(key) for key in keys)
    header = "".join(f"{name:>16s}" for name in names)
    print(f"{'metric':{width}s}{header}{'delta':>12s}")
    for key in keys:
        values = [results[name].get(key, float('nan')) for name in names]
        delta = ""
        if len(values) == 2 and values[0]:
            delta = f"{100 * (values[1] - values[0]) / values[0]:+11.1f}%"
        print(f"{key:{width}s}" + "".join(f"{v:16.3f}" for v in values) + delta)

    if args.output:
        args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
