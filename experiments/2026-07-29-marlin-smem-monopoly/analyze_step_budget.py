#!/usr/bin/env python3
"""Correlation-attributed additive step budget and per-layer EP skew.

`analyze_profile_ab.py` assigns GPU kernels to an engine step by comparing GPU
timestamps against the CPU `execute_` annotation boundaries. Asynchronous launch
means a step's kernels run past that boundary, so its per-family aggregates drop
4-6% of the kernels and score the holes as GPU-empty time. This script assigns
every kernel through its CUDA launch correlation instead, the same way
`analyze_comms.py` does, and fails closed on the resulting per-step kernel
counts.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import statistics
from pathlib import Path

from analyze_profile_ab import family

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATEGORIES = {"cuda_runtime", "cuda_driver"}
RANK_RE = re.compile(r"_rank(\d+)\.")

BUDGET_BUCKETS = {
    "routed W4 Marlin": "routed experts (W4 Marlin)",
    "dense W4 Marlin": "dense/shared compiled GEMMs",
    "dense/shared GEMM": "dense/shared compiled GEMMs",
    "MTP FP8 GEMM": "dense/shared compiled GEMMs",
    "TP custom all-reduce": "TP/EP communication and synchronization",
    "TP vocabulary NCCL all-gather": "TP/EP communication and synchronization",
    "DCP NCCL": "TP/EP communication and synchronization",
    "FlashMLA": "attention (FlashMLA + DSA + KV)",
    "DSA indexer": "attention (FlashMLA + DSA + KV)",
    "DSA top-k": "attention (FlashMLA + DSA + KV)",
    "KV cache": "attention (FlashMLA + DSA + KV)",
    "MoE routing/sort": "MoE routing, activation and sum",
    "MoE activation": "MoE routing, activation and sum",
    "MoE sum": "MoE routing, activation and sum",
    "Triton glue": "glue, elementwise and uncategorized",
    "memory/glue": "glue, elementwise and uncategorized",
    "other": "glue, elementwise and uncategorized",
}
EMPTY_BUCKET = "GPU-empty host/graph gaps"

# Expected per rank per bounded MTP3 decode step: 157 target plus three
# prep+forward custom all-reduces, four vocabulary all-gathers, and two Marlin
# GEMMs for each of the 75 routed target layers and three MTP draft layers.
EXPECT_COUNTS = {
    "cross_device_reduce_1stage": 166,
    "ncclDevKernel_AllGather": 4,
    "marlin_moe_wna16": 306,
}


def load_trace(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    complete = [event for event in events if event.get("ph") == "X"]
    origin = min(
        event["ts"] for event in complete if event.get("cat") in GPU_CATEGORIES
    )
    for event in complete:
        event["t"] = (event["ts"] - origin) / 1000
    return complete


def rank_from_path(path: Path) -> int:
    match = RANK_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot find rank in {path}")
    return int(match.group(1))


def step_windows(events: list[dict]) -> list[tuple[float, float]]:
    execute = sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and event["name"].startswith("execute_")
        ),
        key=lambda event: event["t"],
    )
    return [(a["t"], b["t"]) for a, b in zip(execute, execute[1:])]


def partition(ops: list[dict]) -> tuple[dict[str, float], float, float, float]:
    """Split the step's GPU span among the families active in each segment.

    Every elementary time segment is shared equally among the distinct families
    resident in it, so the family shares sum exactly to the busy union and the
    union plus the empty time sums exactly to the span.
    """
    edges = set()
    spans = []
    for event in ops:
        start = event["t"]
        end = start + event["dur"] / 1000
        spans.append((start, end, BUDGET_BUCKETS[family(event["name"])]))
        edges.add(start)
        edges.add(end)
    ordered = sorted(edges)
    by_start = sorted(spans, key=lambda item: item[0])

    shares: dict[str, float] = collections.defaultdict(float)
    busy = 0.0
    active: list[tuple[float, str]] = []
    cursor = 0
    for left, right in zip(ordered, ordered[1:]):
        while cursor < len(by_start) and by_start[cursor][0] <= left:
            active.append((by_start[cursor][1], by_start[cursor][2]))
            cursor += 1
        active = [item for item in active if item[0] > left]
        if not active:
            continue
        names = {name for _, name in active}
        width = right - left
        busy += width
        for name in names:
            shares[name] += width / len(names)
    span = ordered[-1] - ordered[0]
    return dict(shares), busy, span, ordered[0]


def layer_spans(target_ops: list[dict], target_end: float) -> list[dict]:
    anchors = [event for event in target_ops if "grouped_topk" in event["name"]]
    rows = []
    for index, anchor in enumerate(anchors):
        end = anchors[index + 1]["t"] if index + 1 < len(anchors) else target_end
        segment = [
            event for event in target_ops if anchor["t"] <= event["t"] < end
        ]
        marlin = [
            event
            for event in segment
            if event["name"].startswith("void marlin_moe_wna16")
        ]
        sums = [event for event in segment if "moe_sum_vec" in event["name"]]
        adds = [
            event
            for event in segment
            if "CUDAFunctor_add<c10::BFloat16>" in event["name"]
        ]
        if len(marlin) != 4 or len(sums) != 2 or not adds:
            continue
        hot_stream = min(adds, key=lambda event: event["t"])["args"]["stream"]
        streams = {event["args"]["stream"] for event in marlin}
        if len(streams) != 2 or hot_stream not in streams:
            continue
        cold_stream = next(iter(streams - {hot_stream}))
        hot = [e for e in marlin if e["args"]["stream"] == hot_stream]
        cold = [e for e in marlin if e["args"]["stream"] == cold_stream]
        if len(hot) != 2 or len(cold) != 2:
            continue
        start = min(event["t"] for event in marlin)
        finish = max(
            event["t"] + event["dur"] / 1000 for event in marlin + sums
        )
        rows.append(
            {
                "ordinal": len(rows),
                "hot_us": sum(event["dur"] for event in hot),
                "cold_us": sum(event["dur"] for event in cold),
                "marlin_sum_us": sum(event["dur"] for event in marlin),
                "span_us": (finish - start) * 1000,
            }
        )
    return rows


def rank_steps(path: Path, strict: bool) -> list[dict]:
    events = load_trace(path)
    kernels_by_correlation: dict[int, list[dict]] = collections.defaultdict(list)
    for event in events:
        if event.get("cat") not in GPU_CATEGORIES:
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None:
            kernels_by_correlation[correlation].append(event)

    launches = [
        event for event in events if event.get("cat") in LAUNCH_CATEGORIES
    ]
    launches.sort(key=lambda event: event["t"])

    steps = []
    for index, (start, end) in enumerate(step_windows(events)):
        correlations = [
            event["args"]["correlation"]
            for event in launches
            if start <= event["t"] < end and "correlation" in event.get("args", {})
        ]
        ops = sorted(
            (
                kernel
                for correlation in correlations
                for kernel in kernels_by_correlation[correlation]
            ),
            key=lambda event: event["t"],
        )
        if not ops:
            continue
        counts = {
            needle: sum(1 for event in ops if needle in event["name"])
            for needle in EXPECT_COUNTS
        }
        if strict and counts != EXPECT_COUNTS:
            raise ValueError(
                f"{path.name} step {index}: kernel census {counts} != "
                f"{EXPECT_COUNTS}"
            )
        shares, busy, span, span_start = partition(ops)

        graph_launches = sorted(
            (
                event
                for event in launches
                if start <= event["t"] < end and "GraphLaunch" in event["name"]
            ),
            key=lambda event: event["t"],
        )
        layers: list[dict] = []
        target = {}
        if graph_launches:
            correlation = graph_launches[0]["args"]["correlation"]
            target_ops = sorted(
                kernels_by_correlation[correlation], key=lambda e: e["t"]
            )
            if target_ops:
                target_end = max(
                    event["t"] + event["dur"] / 1000 for event in target_ops
                )
                target = {
                    "span_ms": target_end - min(e["t"] for e in target_ops),
                }
                layers = layer_spans(target_ops, target_end)

        steps.append(
            {
                "step": index,
                "wall_ms": end - start,
                "gpu_span_ms": span,
                "gpu_busy_ms": busy,
                "gpu_empty_ms": span - busy,
                "lead_in_ms": span_start - start,
                "shares": shares,
                "counts": counts,
                "target": target,
                "layers": layers,
            }
        )
    return steps


def summarize(arm_dir: Path, strict: bool) -> dict:
    traces = sorted(arm_dir.glob("*.trace.json.gz"))
    if len(traces) != 4:
        raise ValueError(f"expected four rank traces in {arm_dir}, found {traces}")
    per_rank = {rank_from_path(path): rank_steps(path, strict) for path in traces}

    flat = [step for steps in per_rank.values() for step in steps]
    total = len(flat)
    buckets: dict[str, float] = collections.defaultdict(float)
    for step in flat:
        for name, value in step["shares"].items():
            buckets[name] += value
        buckets[EMPTY_BUCKET] += step["gpu_empty_ms"]
    budget = {name: value / total for name, value in buckets.items()}
    budget_total = sum(budget.values())

    hot = statistics.fmean(
        sum(row["hot_us"] for row in step["layers"]) / 1000
        for step in flat
        if step["layers"]
    )
    cold = statistics.fmean(
        sum(row["cold_us"] for row in step["layers"]) / 1000
        for step in flat
        if step["layers"]
    )
    layer_span = statistics.fmean(
        sum(row["span_us"] for row in step["layers"]) / 1000
        for step in flat
        if step["layers"]
    )

    common = min(len(steps) for steps in per_rank.values())
    skews = []
    worst: dict[int, list[float]] = collections.defaultdict(list)
    for index in range(common):
        rows = [per_rank[rank][index]["layers"] for rank in sorted(per_rank)]
        if any(len(row) != 75 for row in rows):
            continue
        excess = 0.0
        for ordinal in range(75):
            spans = [row[ordinal]["span_us"] for row in rows]
            gap = max(spans) - statistics.fmean(spans)
            excess += gap
            worst[ordinal].append(gap)
        skews.append(excess / 1000)

    return {
        "arm_dir": str(arm_dir),
        "rank_steps": {str(rank): len(steps) for rank, steps in per_rank.items()},
        "rank_steps_total": total,
        "wall_ms_per_step": statistics.fmean(step["wall_ms"] for step in flat),
        "gpu_span_ms_per_step": statistics.fmean(
            step["gpu_span_ms"] for step in flat
        ),
        "gpu_busy_ms_per_step": statistics.fmean(
            step["gpu_busy_ms"] for step in flat
        ),
        "gpu_empty_ms_per_step": statistics.fmean(
            step["gpu_empty_ms"] for step in flat
        ),
        "wall_minus_span_ms_per_step": statistics.fmean(
            step["wall_ms"] - step["gpu_span_ms"] for step in flat
        ),
        "kernel_census": dict(flat[0]["counts"]),
        "budget_ms_per_step": dict(
            sorted(budget.items(), key=lambda item: -item[1])
        ),
        "budget_total_ms_per_step": budget_total,
        "target_graph_span_ms_per_step": statistics.fmean(
            step["target"]["span_ms"] for step in flat if step["target"]
        ),
        "tiered": {
            "hot_ms_per_step": hot,
            "cold_ms_per_step": cold,
            "marlin_cumulative_ms_per_step": hot + cold,
            "layer_span_ms_per_step": layer_span,
            "overlap_headroom_ms_per_step": layer_span - max(hot, cold),
        },
        "ep_skew": {
            "aligned_steps": len(skews),
            "summed_max_minus_mean_ms_per_step": (
                statistics.fmean(skews) if skews else None
            ),
            "spread_ms": [min(skews), max(skews)] if skews else None,
            "worst_layers_us": sorted(
                (
                    {"layer": ordinal, "mean_excess_us": statistics.fmean(gaps)}
                    for ordinal, gaps in worst.items()
                ),
                key=lambda row: -row["mean_excess_us"],
            )[:8],
        },
    }


def print_arm(label: str, summary: dict) -> None:
    print(f"\n=== {label}: {summary['arm_dir']}")
    print(
        f"rank-steps {summary['rank_steps_total']}  census {summary['kernel_census']}"
    )
    for key in (
        "wall_ms_per_step",
        "gpu_span_ms_per_step",
        "gpu_busy_ms_per_step",
        "gpu_empty_ms_per_step",
        "wall_minus_span_ms_per_step",
        "target_graph_span_ms_per_step",
    ):
        print(f"  {key:34s} {summary[key]:8.3f}")
    print("  additive budget:")
    total = summary["budget_total_ms_per_step"]
    for name, value in summary["budget_ms_per_step"].items():
        print(f"    {name:44s} {value:7.3f} ms  {100 * value / total:5.2f}%")
    print(f"    {'total':44s} {total:7.3f} ms")
    tiered = summary["tiered"]
    print(
        f"  hot {tiered['hot_ms_per_step']:.3f}  cold {tiered['cold_ms_per_step']:.3f}"
        f"  marlin {tiered['marlin_cumulative_ms_per_step']:.3f}"
        f"  span {tiered['layer_span_ms_per_step']:.3f}"
        f"  headroom {tiered['overlap_headroom_ms_per_step']:.3f}"
    )
    skew = summary["ep_skew"]
    print(
        f"  EP skew {skew['summed_max_minus_mean_ms_per_step']:.3f} ms/step over "
        f"{skew['aligned_steps']} aligned steps, spread "
        f"{skew['spread_ms'][0]:.3f}-{skew['spread_ms'][1]:.3f}"
    )
    print(
        "  worst layers: "
        + ", ".join(
            f"L{row['layer']} {row['mean_excess_us']:.0f}us"
            for row in skew["worst_layers_us"][:5]
        )
    )


def cross_capture(arms: list[dict]) -> dict:
    def mean(select) -> float:
        return statistics.fmean(select(arm) for arm in arms)

    names = arms[0]["budget_ms_per_step"]
    total = mean(lambda arm: arm["budget_total_ms_per_step"])
    return {
        "captures": len(arms),
        "wall_ms_per_step": mean(lambda arm: arm["wall_ms_per_step"]),
        "gpu_span_ms_per_step": total,
        "budget": {
            name: {
                "ms_per_step": mean(lambda arm: arm["budget_ms_per_step"][name]),
                "percent": 100
                * mean(lambda arm: arm["budget_ms_per_step"][name])
                / total,
                "spread_ms": [
                    min(arm["budget_ms_per_step"][name] for arm in arms),
                    max(arm["budget_ms_per_step"][name] for arm in arms),
                ],
            }
            for name in sorted(
                names, key=lambda name: -mean(lambda arm: arm["budget_ms_per_step"][name])
            )
        },
        "tiered": {
            key: {
                "mean": mean(lambda arm: arm["tiered"][key]),
                "spread": [
                    min(arm["tiered"][key] for arm in arms),
                    max(arm["tiered"][key] for arm in arms),
                ],
            }
            for key in arms[0]["tiered"]
        },
        "ep_skew_ms_per_step": {
            "mean": mean(
                lambda arm: arm["ep_skew"]["summed_max_minus_mean_ms_per_step"]
            ),
            "spread": [
                min(
                    arm["ep_skew"]["summed_max_minus_mean_ms_per_step"]
                    for arm in arms
                ),
                max(
                    arm["ep_skew"]["summed_max_minus_mean_ms_per_step"]
                    for arm in arms
                ),
            ],
        },
    }


def print_cross(label: str, summary: dict) -> None:
    print(f"\n=== {label}, mean over {summary['captures']} captures")
    print(
        f"  step wall {summary['wall_ms_per_step']:.3f} ms, attributed GPU span "
        f"{summary['gpu_span_ms_per_step']:.3f} ms"
    )
    for name, row in summary["budget"].items():
        print(
            f"    {name:44s} {row['ms_per_step']:6.3f} ms  {row['percent']:5.2f}%"
            f"  [{row['spread_ms'][0]:.3f}, {row['spread_ms'][1]:.3f}]"
        )
    for key, row in summary["tiered"].items():
        print(
            f"  {key:36s} {row['mean']:6.3f}  "
            f"[{row['spread'][0]:.3f}, {row['spread'][1]:.3f}]"
        )
    skew = summary["ep_skew_ms_per_step"]
    print(
        f"  EP skew (summed max-minus-mean)      {skew['mean']:6.3f}  "
        f"[{skew['spread'][0]:.3f}, {skew['spread'][1]:.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="report the per-step kernel census instead of requiring it",
    )
    args = parser.parse_args()

    results = {}
    for capture in args.captures:
        name = capture.name.rsplit("-", 1)[-1]
        for arm in ("off", "on"):
            summary = summarize(capture / arm, not args.no_strict)
            results[f"{name}/{arm}"] = summary
            print_arm(f"{name}/{arm}", summary)

    for arm in ("off", "on"):
        arms = [
            summary for key, summary in results.items() if key.endswith(f"/{arm}")
        ]
        if len(arms) > 1:
            results[f"cross-capture/{arm}"] = cross_capture(arms)
            print_cross(arm, results[f"cross-capture/{arm}"])

    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
