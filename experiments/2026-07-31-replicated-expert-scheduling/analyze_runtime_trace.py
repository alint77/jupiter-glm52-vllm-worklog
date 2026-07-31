#!/usr/bin/env python3
"""Compare steady c4 decode graphs with replica assignment off and greedy."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import statistics
from pathlib import Path

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
RUNTIME_CATEGORIES = {"cuda_runtime", "cuda_driver"}
STEP_MARKER = "_generation_4(16)"
RANK_RE = re.compile(r"_rank(\d+)\.")


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [
            event
            for event in json.load(handle)["traceEvents"]
            if event.get("ph") == "X"
        ]


def family(name: str) -> str:
    if "assign_replicated_experts" in name:
        return "replica scheduler"
    if "marlin_moe_wna16" in name:
        return "routed W4 Marlin"
    if "cross_device_reduce_1stage" in name:
        return "TP custom all-reduce"
    if "ncclDevKernel_AllGather" in name:
        return "NCCL all-gather"
    return "other"


def union_us(events: list[dict]) -> float:
    spans = sorted((event["ts"], event["ts"] + event["dur"]) for event in events)
    merged: list[list[float]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def layer_spans(ops: list[dict]) -> list[float]:
    anchors = sorted(
        (event for event in ops if "grouped_topk" in event["name"]),
        key=lambda event: event["ts"],
    )
    graph_end = max(event["ts"] + event["dur"] for event in ops)
    rows = []
    for index, anchor in enumerate(anchors):
        end = anchors[index + 1]["ts"] if index + 1 < len(anchors) else graph_end
        segment = [event for event in ops if anchor["ts"] <= event["ts"] < end]
        marlin = [event for event in segment if "marlin_moe_wna16" in event["name"]]
        moe_sum = [event for event in segment if "moe_sum_vec" in event["name"]]
        if len(marlin) != 4 or len(moe_sum) != 2:
            continue
        start = min(event["ts"] for event in marlin)
        finish = max(event["ts"] + event["dur"] for event in marlin + moe_sum)
        rows.append(finish - start)
    return rows


def rank_steps(path: Path) -> list[dict]:
    events = load(path)
    kernels_by_correlation: dict[int, list[dict]] = collections.defaultdict(list)
    for event in events:
        correlation = event.get("args", {}).get("correlation")
        if event.get("cat") in GPU_CATEGORIES and correlation is not None:
            kernels_by_correlation[correlation].append(event)

    runtime = [event for event in events if event.get("cat") in RUNTIME_CATEGORIES]
    annotations = sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation" and STEP_MARKER in event["name"]
        ),
        key=lambda event: event["ts"],
    )
    steps = []
    for annotation in annotations:
        end = annotation["ts"] + annotation["dur"]
        launches = [
            event
            for event in runtime
            if annotation["ts"] <= event["ts"] < end and "GraphLaunch" in event["name"]
        ]
        if len(launches) != 1:
            raise ValueError(
                f"{path.name}: expected one decode graph, found {len(launches)}"
            )
        ops = sorted(
            kernels_by_correlation[launches[0]["args"]["correlation"]],
            key=lambda event: event["ts"],
        )
        if not ops:
            raise ValueError(f"{path.name}: decode graph has no device operations")
        families: dict[str, list[dict]] = collections.defaultdict(list)
        for event in ops:
            families[family(event["name"])].append(event)
        steps.append(
            {
                "annotation_start_us": annotation["ts"],
                "graph_start_us": min(event["ts"] for event in ops),
                "graph_span_us": max(event["ts"] + event["dur"] for event in ops)
                - min(event["ts"] for event in ops),
                "gpu_busy_us": union_us(ops),
                "families": {
                    name: {
                        "cumulative_us": sum(event["dur"] for event in members),
                        "calls": len(members),
                    }
                    for name, members in families.items()
                },
                "layer_spans_us": layer_spans(ops),
            }
        )
    return steps


def summarize(directory: Path) -> dict:
    traces = sorted(directory.glob("*.trace.json.gz"))
    if len(traces) != 4:
        raise ValueError(f"expected four rank traces in {directory}")
    per_rank = {}
    for path in traces:
        match = RANK_RE.search(path.name)
        if match is None:
            raise ValueError(f"cannot read rank from {path.name}")
        per_rank[int(match.group(1))] = rank_steps(path)

    flat = [step for steps in per_rank.values() for step in steps]
    if any(len(step["layer_spans_us"]) != 75 for step in flat):
        raise ValueError("not every profiled graph contains 75 routed layers")
    family_names = {
        name for step in flat for name in step["families"] if name != "other"
    }

    graph_cycles = []
    annotation_cycles = []
    for steps in per_rank.values():
        graph_cycles.extend(
            right["graph_start_us"] - left["graph_start_us"]
            for left, right in zip(steps, steps[1:])
        )
        annotation_cycles.extend(
            right["annotation_start_us"] - left["annotation_start_us"]
            for left, right in zip(steps, steps[1:])
        )

    skew = []
    for step_index in range(min(map(len, per_rank.values()))):
        layer_rows = [
            per_rank[rank][step_index]["layer_spans_us"] for rank in sorted(per_rank)
        ]
        skew.append(
            sum(max(values) - statistics.fmean(values) for values in zip(*layer_rows))
        )

    return {
        "trace_dir": str(directory),
        "rank_steps": {str(rank): len(steps) for rank, steps in per_rank.items()},
        "rank_steps_total": len(flat),
        "graph_span_ms_per_rank_step": statistics.fmean(
            step["graph_span_us"] for step in flat
        )
        / 1000,
        "graph_start_cycle_ms": statistics.fmean(graph_cycles) / 1000,
        "annotation_start_cycle_ms": statistics.fmean(annotation_cycles) / 1000,
        "gpu_busy_ms_per_rank_step": statistics.fmean(
            step["gpu_busy_us"] for step in flat
        )
        / 1000,
        "summed_layer_max_minus_mean_ms": statistics.fmean(skew) / 1000,
        "kernel_families": {
            name: {
                "cumulative_ms_per_rank_step": statistics.fmean(
                    step["families"].get(name, {}).get("cumulative_us", 0)
                    for step in flat
                )
                / 1000,
                "calls_per_rank_step": statistics.fmean(
                    step["families"].get(name, {}).get("calls", 0) for step in flat
                ),
            }
            for name in sorted(family_names)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("off", type=Path)
    parser.add_argument("greedy", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    results = {"off": summarize(args.off), "greedy": summarize(args.greedy)}
    for name, summary in results.items():
        print(f"\n{name}: {summary['rank_steps_total']} rank-steps")
        print(
            f"  graph cycle {summary['graph_start_cycle_ms']:.3f} ms, "
            f"graph span {summary['graph_span_ms_per_rank_step']:.3f} ms"
        )
        print(
            "  summed layer max-minus-mean "
            f"{summary['summed_layer_max_minus_mean_ms']:.3f} ms"
        )
        for family_name, row in summary["kernel_families"].items():
            print(
                f"  {family_name:24s} "
                f"{row['cumulative_ms_per_rank_step']:.3f} ms, "
                f"{row['calls_per_rank_step']:.0f} calls"
            )
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
