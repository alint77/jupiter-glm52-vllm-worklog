#!/usr/bin/env python3
"""Locate a decode step's GPU-empty time relative to its CUDA graph replays.

The step budget reports GPU-empty time as one bucket. That bucket has two very
different halves: dependency gaps *inside* a graph replay, which no host change
can remove, and gaps *between* replays, where the device waits on host work. This
script splits them, attributes the between-graph half to the specific graph
boundary it follows, and reports whether the host was inside a CUDA API during
the gap or doing something the profiler does not see as a launch.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATEGORIES = {"cuda_runtime", "cuda_driver"}


def load_trace(path: Path) -> list[dict]:
    events = [
        event
        for event in json.load(gzip.open(path, "rt"))["traceEvents"]
        if event.get("ph") == "X"
    ]
    origin = min(
        event["ts"] for event in events if event.get("cat") in GPU_CATEGORIES
    )
    for event in events:
        event["t"] = (event["ts"] - origin) / 1000
    return events


def merge(ops: list[dict]) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(
        (event["t"], event["t"] + event["dur"] / 1000) for event in ops
    ):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def summarize(arm_dir: Path) -> dict:
    in_graph: collections.Counter[int] = collections.Counter()
    boundary: collections.Counter[int] = collections.Counter()
    eager: collections.Counter[str] = collections.Counter()
    graph_kernels: collections.Counter[int] = collections.Counter()
    host_busy = 0.0
    host_dark = 0.0
    steps = 0

    for path in sorted(arm_dir.glob("*.trace.json.gz")):
        events = load_trace(path)
        by_correlation: dict[int, list[dict]] = collections.defaultdict(list)
        for event in events:
            if event.get("cat") not in GPU_CATEGORIES:
                continue
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                by_correlation[correlation].append(event)
        runtime = sorted(
            (e for e in events if e.get("cat") in LAUNCH_CATEGORIES),
            key=lambda event: event["t"],
        )
        launches = [e for e in runtime if "correlation" in e.get("args", {})]
        execute = sorted(
            (
                e
                for e in events
                if e.get("cat") == "user_annotation"
                and e["name"].startswith("execute_")
            ),
            key=lambda event: event["t"],
        )
        for start, end in zip(
            (e["t"] for e in execute), (e["t"] for e in execute[1:])
        ):
            window = [e for e in launches if start <= e["t"] < end]
            spans = []
            for launch in window:
                if "GraphLaunch" not in launch["name"]:
                    continue
                kernels = by_correlation[launch["args"]["correlation"]]
                if kernels:
                    spans.append(
                        (
                            min(k["t"] for k in kernels),
                            max(k["t"] + k["dur"] / 1000 for k in kernels),
                            len(kernels),
                        )
                    )
            spans.sort()
            ops = sorted(
                (k for e in window for k in by_correlation[e["args"]["correlation"]]),
                key=lambda event: event["t"],
            )
            if not ops or not spans:
                continue
            steps += 1
            for index, span in enumerate(spans):
                graph_kernels[index] += span[2]

            in_window = [e for e in runtime if start <= e["t"] < end]
            merged = merge(ops)
            for (_, gap_start), (gap_end, _) in zip(merged, merged[1:]):
                width = gap_end - gap_start
                inside = next(
                    (
                        index
                        for index, (left, right, _) in enumerate(spans)
                        if left <= gap_start and gap_end <= right
                    ),
                    None,
                )
                if inside is not None:
                    in_graph[inside] += width
                    continue
                boundary[sum(1 for _, right, _ in spans if right <= gap_start + 1e-9)] += width
                covered = 0.0
                for event in in_window:
                    low = max(event["t"], gap_start)
                    high = min(event["t"] + event["dur"] / 1000, gap_end)
                    if high > low:
                        covered += high - low
                host_busy += min(covered, width)
                host_dark += max(width - covered, 0.0)
            for kernel in ops:
                if not any(left <= kernel["t"] < right for left, right, _ in spans):
                    eager[kernel["name"][:64]] += kernel["dur"] / 1000

    return {
        "arm_dir": str(arm_dir),
        "rank_steps": steps,
        "graphs_per_step": len(graph_kernels),
        "kernels_per_graph": {
            str(index): count / steps for index, count in sorted(graph_kernels.items())
        },
        "in_graph_gap_ms_per_step": {
            str(index): value / steps for index, value in sorted(in_graph.items())
        },
        "in_graph_gap_total_ms_per_step": sum(in_graph.values()) / steps,
        "between_graph_gap_ms_per_step": sum(boundary.values()) / steps,
        "between_graph_gap_by_boundary_ms_per_step": {
            str(index): value / steps for index, value in sorted(boundary.items())
        },
        "host_inside_cuda_api_ms_per_step": host_busy / steps,
        "host_not_in_cuda_api_ms_per_step": host_dark / steps,
        "ungraphed_kernels_ms_per_step": {
            name: value / steps for name, value in eager.most_common(10)
        },
        "ungraphed_total_ms_per_step": sum(eager.values()) / steps,
    }


def print_arm(label: str, summary: dict) -> None:
    print(f"\n=== {label}  ({summary['rank_steps']} rank-steps)")
    print(
        f"  {summary['graphs_per_step']} graph replays/step, kernels: "
        + ", ".join(
            f"{count:.0f}" for count in summary["kernels_per_graph"].values()
        )
    )
    print(
        f"  gaps inside graph replays   {summary['in_graph_gap_total_ms_per_step']:.3f} ms/step"
    )
    for index, value in summary["in_graph_gap_ms_per_step"].items():
        print(f"    graph {index}: {value:.4f}")
    print(
        f"  gaps between graph replays  {summary['between_graph_gap_ms_per_step']:.3f} ms/step"
    )
    for index, value in summary["between_graph_gap_by_boundary_ms_per_step"].items():
        print(f"    after graph {index}: {value:.4f}")
    print(
        f"    host inside a CUDA API    {summary['host_inside_cuda_api_ms_per_step']:.3f}"
    )
    print(
        f"    host not in a CUDA API    {summary['host_not_in_cuda_api_ms_per_step']:.3f}"
    )
    print(
        f"  kernels outside every graph {summary['ungraphed_total_ms_per_step']:.3f} ms/step"
    )
    for name, value in list(summary["ungraphed_kernels_ms_per_step"].items())[:5]:
        print(f"    {value:.4f}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = {}
    for capture in args.captures:
        name = capture.name.rsplit("-", 1)[-1]
        for arm in ("off", "on"):
            summary = summarize(capture / arm)
            results[f"{name}/{arm}"] = summary
            print_arm(f"{name}/{arm}", summary)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
