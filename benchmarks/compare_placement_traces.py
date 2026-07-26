#!/usr/bin/env python3

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def trace_path(trace_dir: Path, rank: int) -> Path:
    matches = [
        path
        for path in trace_dir.iterdir()
        if f"rank{rank}." in path.name
        and (path.name.endswith(".json") or path.name.endswith(".json.gz"))
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one rank-{rank} trace in {trace_dir}: {matches}")
    return matches[0]


def load_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as file:
        return json.load(file)["traceEvents"]


def kernel_category(name: str) -> str:
    lower = name.lower()
    if "marlin_moe_wna16" in lower:
        return "marlin_moe"
    if any(token in lower for token in ("act_and_mul", "moe_sum_vec")):
        return "moe_aux"
    if any(
        token in lower
        for token in (
            "nccl",
            "allreduce",
            "all_reduce",
            "cross_device_reduce",
            "reduce_scatter",
            "all_gather",
            "msccl",
            "nvls",
        )
    ):
        return "collective"
    if any(token in lower for token in ("flash_mla", "flashmla", "fmha")):
        return "attention"
    if any(token in lower for token in ("deep_gemm", "deepgemm")):
        return "deep_gemm"
    if any(token in lower for token in ("grouped_topk", "moe_align", "topk")):
        return "routing"
    if any(token in lower for token in ("gemm", "cutlass", "cublas")):
        return "other_gemm"
    if "memcpy" in lower:
        return "memcpy"
    if "memset" in lower:
        return "memset"
    return "other"


def union_and_gaps(events: list[dict]) -> tuple[float, list[float], float]:
    intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in events)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    busy = sum(end - start for start, end in merged)
    gaps = [
        merged[index + 1][0] - merged[index][1]
        for index in range(len(merged) - 1)
    ]
    span = merged[-1][1] - merged[0][0]
    return busy, gaps, span


def tier_chains(kernels: list[dict]) -> dict:
    anchors = sorted(
        (event for event in kernels if "grouped_topk" in event["name"]),
        key=lambda event: event["ts"],
    )
    rows = []
    for index, anchor in enumerate(anchors):
        end = anchors[index + 1]["ts"] if index + 1 < len(anchors) else float("inf")
        segment = [
            event for event in kernels if anchor["ts"] <= event["ts"] < end
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
        hot_stream = min(adds, key=lambda event: event["ts"])["args"]["stream"]
        streams = {event["args"]["stream"] for event in marlin}
        if hot_stream not in streams or len(streams) != 2:
            continue
        cold_stream = next(iter(streams - {hot_stream}))
        hot = sorted(
            (event for event in marlin if event["args"]["stream"] == hot_stream),
            key=lambda event: event["ts"],
        )
        cold = sorted(
            (event for event in marlin if event["args"]["stream"] == cold_stream),
            key=lambda event: event["ts"],
        )
        if len(hot) != 2 or len(cold) != 2:
            continue
        hot_end = hot[-1]["ts"] + hot[-1]["dur"]
        cold_end = cold[-1]["ts"] + cold[-1]["dur"]
        span_start = min(event["ts"] for event in marlin)
        span_end = max(
            event["ts"] + event["dur"] for event in marlin + sums
        )
        rows.append(
            {
                "hot_w13_us": hot[0]["dur"],
                "hot_w2_us": hot[1]["dur"],
                "cold_w13_us": cold[0]["dur"],
                "cold_w2_us": cold[1]["dur"],
                "hot_us": sum(event["dur"] for event in hot),
                "cold_us": sum(event["dur"] for event in cold),
                "span_us": span_end - span_start,
                "marlin_sum_us": sum(event["dur"] for event in marlin),
                "cold_finished_first": cold_end < hot_end,
            }
        )
    if not rows:
        return {"recognized_layers": 0}
    marlin_sum = sum(row["marlin_sum_us"] for row in rows)
    span = sum(row["span_us"] for row in rows)
    return {
        "recognized_layers": len(rows),
        "hot_w13_ms": sum(row["hot_w13_us"] for row in rows) / 1000,
        "hot_w2_ms": sum(row["hot_w2_us"] for row in rows) / 1000,
        "cold_w13_ms": sum(row["cold_w13_us"] for row in rows) / 1000,
        "cold_w2_ms": sum(row["cold_w2_us"] for row in rows) / 1000,
        "hot_kernel_time_ms": sum(row["hot_us"] for row in rows) / 1000,
        "cold_kernel_time_ms": sum(row["cold_us"] for row in rows) / 1000,
        "layer_span_ms": span / 1000,
        "marlin_kernel_time_ms": marlin_sum / 1000,
        "tier_overlap_percent": 100 * (1 - span / marlin_sum),
        "cold_finished_first_percent": 100
        * statistics.mean(row["cold_finished_first"] for row in rows),
    }


def summarize_rank(path: Path) -> dict:
    events = load_trace(path)
    kernels = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") in GPU_CATEGORIES
    ]
    if not kernels:
        raise ValueError(f"No GPU events in {path}")
    busy, gaps, span = union_and_gaps(kernels)
    category_time = defaultdict(float)
    category_count = defaultdict(int)
    exact_time = defaultdict(float)
    for event in kernels:
        category = kernel_category(event["name"])
        category_time[category] += event["dur"] / 1000
        category_count[category] += 1
        exact_time[event["name"]] += event["dur"] / 1000
    gaps_sorted = sorted(gaps)
    p95_index = max(0, int(0.95 * len(gaps_sorted)) - 1)
    return {
        "trace_file": str(path),
        "gpu_span_ms": span / 1000,
        "gpu_busy_ms": busy / 1000,
        "gpu_idle_percent": 100 * (1 - busy / span),
        "kernel_count": len(kernels),
        "gap_count": len(gaps),
        "mean_gap_us": statistics.mean(gaps) if gaps else 0,
        "p95_gap_us": gaps_sorted[p95_index] if gaps else 0,
        "maximum_gap_us": max(gaps, default=0),
        "category_kernel_time_ms": dict(sorted(category_time.items())),
        "category_kernel_count": dict(sorted(category_count.items())),
        "top_kernels": [
            {"name": name, "kernel_time_ms": duration}
            for name, duration in sorted(
                exact_time.items(), key=lambda item: -item[1]
            )[:20]
        ],
        "tier_chains": tier_chains(kernels),
    }


def mean_rank_metric(ranks: list[dict], key: str) -> float:
    return statistics.mean(rank[key] for rank in ranks)


def summarize(trace_dir: Path) -> dict:
    ranks = [summarize_rank(trace_path(trace_dir, rank)) for rank in range(4)]
    categories = sorted(
        {
            category
            for rank in ranks
            for category in rank["category_kernel_time_ms"]
        }
    )
    tier_keys = (
        "hot_w13_ms",
        "hot_w2_ms",
        "cold_w13_ms",
        "cold_w2_ms",
        "hot_kernel_time_ms",
        "cold_kernel_time_ms",
        "layer_span_ms",
        "marlin_kernel_time_ms",
        "tier_overlap_percent",
        "cold_finished_first_percent",
    )
    return {
        "trace_dir": str(trace_dir),
        "rank_mean": {
            key: mean_rank_metric(ranks, key)
            for key in (
                "gpu_span_ms",
                "gpu_busy_ms",
                "gpu_idle_percent",
                "kernel_count",
                "mean_gap_us",
                "p95_gap_us",
                "maximum_gap_us",
            )
        },
        "rank_mean_category_kernel_time_ms": {
            category: statistics.mean(
                rank["category_kernel_time_ms"].get(category, 0)
                for rank in ranks
            )
            for category in categories
        },
        "rank_mean_tier_chains": {
            key: statistics.mean(rank["tier_chains"][key] for rank in ranks)
            for key in tier_keys
        },
        "ranks": ranks,
    }


def percent_delta(candidate: float, control: float) -> float:
    return 100 * (candidate / control - 1) if control else 0


def main() -> None:
    args = parse_args()
    control = summarize(args.control_dir)
    candidate = summarize(args.candidate_dir)
    comparison = {}
    for group in (
        "rank_mean",
        "rank_mean_category_kernel_time_ms",
        "rank_mean_tier_chains",
    ):
        comparison[group] = {
            key: percent_delta(candidate[group][key], control[group][key])
            for key in control[group].keys() & candidate[group].keys()
        }
    report = {
        "control": control,
        "candidate": candidate,
        "candidate_delta_percent": comparison,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
