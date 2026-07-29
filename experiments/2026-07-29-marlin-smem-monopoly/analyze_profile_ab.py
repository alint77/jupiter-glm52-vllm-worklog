#!/usr/bin/env python3
"""Compare matched PyTorch profiler traces with tight Marlin smem off/on."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def load_trace(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    first_gpu = min(
        event["ts"]
        for event in events
        if event.get("ph") == "X" and event.get("cat") in GPU_CATEGORIES
    )
    for event in events:
        if event.get("ph") == "X":
            event["t"] = (event["ts"] - first_gpu) / 1000
    return events


def step_bounds(events: list[dict]) -> list[tuple[float, float]]:
    execute = sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and event["name"].startswith("execute_")
        ),
        key=lambda event: event["t"],
    )
    return [(execute[i]["t"], execute[i + 1]["t"]) for i in range(len(execute) - 1)]


def gpu_ops(events: list[dict], start: float, end: float) -> list[dict]:
    return sorted(
        (
            event
            for event in events
            if event.get("cat") in GPU_CATEGORIES and start <= event["t"] < end
        ),
        key=lambda event: event["t"],
    )


def intervals(ops: list[dict]) -> list[list[float]]:
    spans = sorted(
        (event["t"], event["t"] + event["dur"] / 1000) for event in ops
    )
    merged: list[list[float]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def union_ms(ops: list[dict]) -> float:
    return sum(end - start for start, end in intervals(ops))


def family(name: str) -> str:
    if name.startswith("triton_") and "fused_mm" in name:
        return "dense/shared GEMM"
    families = (
        ("routed W4 Marlin", "marlin_moe_wna16"),
        ("TP custom all-reduce", "cross_device_reduce"),
        ("TP vocabulary NCCL all-gather", "ncclDevKernel"),
        ("MoE activation", "act_and_mul"),
        ("MoE sum", "moe_sum_vec"),
        ("MoE routing/sort", "grouped_topk"),
        ("MoE routing/sort", "count_and_sort_expert"),
        ("MoE routing/sort", "moe_align_block_size"),
        ("dense W4 Marlin", "marlin::Marlin"),
        ("dense/shared GEMM", "cutlass::device_kernel"),
        ("dense/shared GEMM", "nvjet_"),
        ("FlashMLA", "sparse_fp8::flash_fwd"),
        ("FlashMLA", "flash_fwd_mla_combine"),
        ("DSA indexer", "deep_gemm::sm90_fp8"),
        ("DSA top-k", "cooperative_topk"),
        ("DSA top-k", "StableTopK"),
        ("MTP FP8 GEMM", "deep_gemm::fp8_gemm"),
        ("KV cache", "concat_and_cache"),
        ("KV cache", "indexer_k_quant"),
        ("Triton glue", "triton_"),
        ("memory/glue", "elementwise_kernel"),
        ("memory/glue", "Memset"),
        ("memory/glue", "Memcpy"),
        ("memory/glue", "memcpy32_post"),
    )
    for label, needle in families:
        if needle in name:
            return label
    return "other"


def layer_rows(ops: list[dict], step_end: float) -> list[dict]:
    anchors = [event for event in ops if "grouped_topk" in event["name"]]
    rows = []
    for index, anchor in enumerate(anchors):
        end = anchors[index + 1]["t"] if index + 1 < len(anchors) else step_end
        segment = [event for event in ops if anchor["t"] <= event["t"] < end]
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
        hot = sorted(
            (event for event in marlin if event["args"]["stream"] == hot_stream),
            key=lambda event: event["t"],
        )
        cold = sorted(
            (event for event in marlin if event["args"]["stream"] == cold_stream),
            key=lambda event: event["t"],
        )
        if len(hot) != 2 or len(cold) != 2:
            continue
        start = min(event["t"] for event in marlin)
        finish = max(
            event["t"] + event["dur"] / 1000 for event in marlin + sums
        )
        rows.append(
            {
                "hot_us": sum(event["dur"] for event in hot),
                "cold_us": sum(event["dur"] for event in cold),
                "marlin_sum_us": sum(event["dur"] for event in marlin),
                "span_us": (finish - start) * 1000,
            }
        )
    return rows


def summarize_arm(trace_dir: Path) -> dict:
    aggregate = collections.Counter()
    category_cumulative = collections.Counter()
    category_union = collections.Counter()
    category_count = collections.Counter()
    exact_kernels = collections.Counter()
    exact_counts = collections.Counter()
    idle_successor = collections.Counter()
    layer_totals = collections.Counter()
    rank_steps = collections.Counter()

    traces = sorted(trace_dir.glob("*.trace.json.gz"))
    if len(traces) != 4:
        raise ValueError(f"expected four rank traces in {trace_dir}, found {traces}")

    for trace in traces:
        events = load_trace(trace)
        all_gpu = [
            event for event in events if event.get("cat") in GPU_CATEGORIES
        ]
        bounds = step_bounds(events)
        if not bounds:
            raise ValueError(f"no steady M=4 steps found in {trace}")
        for start, end in bounds:
            ops = gpu_ops(events, start, end)
            if not ops:
                continue
            aggregate["steps"] += 1
            aggregate["wall_ms"] += end - start
            aggregate["gpu_busy_ms"] += union_ms(ops)
            rank_steps[trace.name.split("_rank")[-1].split(".")[0]] += 1

            buckets: dict[str, list[dict]] = collections.defaultdict(list)
            for event in ops:
                label = family(event["name"])
                buckets[label].append(event)
                exact_kernels[event["name"]] += event["dur"] / 1000
                exact_counts[event["name"]] += 1
            for label, bucket in buckets.items():
                category_cumulative[label] += sum(
                    event["dur"] for event in bucket
                ) / 1000
                category_union[label] += union_ms(bucket)
                category_count[label] += len(bucket)

            merged = intervals(ops)
            next_index = 0
            for gap_start, gap_end in zip(
                [start, *(item[1] for item in merged)],
                [*(item[0] for item in merged), end],
            ):
                if gap_end <= gap_start:
                    continue
                while (
                    next_index < len(ops)
                    and ops[next_index]["t"] < gap_end - 1e-9
                ):
                    next_index += 1
                successor = (
                    family(ops[next_index]["name"])
                    if next_index < len(ops)
                    else "end of step"
                )
                idle_successor[successor] += gap_end - gap_start

            runtime = [
                event
                for event in events
                if event.get("cat") in {"cuda_runtime", "cuda_driver"}
                and start <= event["t"] < end
            ]
            graph_launches = sorted(
                (
                    event
                    for event in runtime
                    if "GraphLaunch" in event["name"]
                ),
                key=lambda event: event["t"],
            )
            kernel_launches = [
                event
                for event in runtime
                if "LaunchKernel" in event["name"]
                or "launchKernel" in event["name"]
            ]
            aggregate["graph_launches"] += len(graph_launches)
            aggregate["graph_launch_host_ms"] += sum(
                event["dur"] for event in graph_launches
            ) / 1000
            aggregate["kernel_launches"] += len(kernel_launches)
            aggregate["kernel_launch_host_ms"] += sum(
                event["dur"] for event in kernel_launches
            ) / 1000
            if graph_launches:
                target_correlation = graph_launches[0]["args"]["correlation"]
                target_ops = [
                    event
                    for event in all_gpu
                    if event.get("args", {}).get("correlation")
                    == target_correlation
                ]
                if target_ops:
                    target_start = min(event["t"] for event in target_ops)
                    target_end = max(
                        event["t"] + event["dur"] / 1000
                        for event in target_ops
                    )
                    aggregate["target_graph_ms"] += target_end - target_start
                    aggregate["target_graph_busy_ms"] += union_ms(target_ops)
                    rows = layer_rows(target_ops, target_end)
                    layer_totals["layers"] += len(rows)
                    for row in rows:
                        layer_totals.update(row)

    steps = aggregate["steps"]
    summary = {
        "trace_dir": str(trace_dir),
        "trace_files": [trace.name for trace in traces],
        "rank_steps": dict(rank_steps),
        "rank_steps_total": steps,
        "wall_ms_per_step": aggregate["wall_ms"] / steps,
        "gpu_busy_ms_per_step": aggregate["gpu_busy_ms"] / steps,
        "gpu_idle_ms_per_step": (
            aggregate["wall_ms"] - aggregate["gpu_busy_ms"]
        )
        / steps,
        "graph_launches_per_step": aggregate["graph_launches"] / steps,
        "graph_launch_host_ms_per_step": aggregate["graph_launch_host_ms"] / steps,
        "kernel_launches_per_step": aggregate["kernel_launches"] / steps,
        "kernel_launch_host_ms_per_step": aggregate["kernel_launch_host_ms"] / steps,
        "target_graph_ms_per_step": aggregate["target_graph_ms"] / steps,
        "target_graph_busy_ms_per_step": (
            aggregate["target_graph_busy_ms"] / steps
        ),
        "categories": {
            label: {
                "cumulative_ms_per_step": category_cumulative[label] / steps,
                "union_ms_per_step": category_union[label] / steps,
                "calls_per_step": category_count[label] / steps,
            }
            for label in category_cumulative
        },
        "idle_by_successor_ms_per_step": {
            label: duration / steps
            for label, duration in idle_successor.most_common()
        },
        "top_kernels": [
            {
                "name": name,
                "cumulative_ms_per_step": duration / steps,
                "calls_per_step": exact_counts[name] / steps,
            }
            for name, duration in exact_kernels.most_common(20)
        ],
    }
    layers = layer_totals["layers"]
    if layers:
        summary["tiered_layers"] = {
            "layers_per_step": layers / steps,
            "hot_ms_per_step": layer_totals["hot_us"] / 1000 / steps,
            "cold_ms_per_step": layer_totals["cold_us"] / 1000 / steps,
            "marlin_sum_ms_per_step": (
                layer_totals["marlin_sum_us"] / 1000 / steps
            ),
            "layer_span_ms_per_step": layer_totals["span_us"] / 1000 / steps,
            "timeline_overlap_percent": 100
            * (
                1
                - layer_totals["span_us"]
                / layer_totals["marlin_sum_us"]
            ),
        }
    return summary


def print_report(results: dict) -> None:
    off, on = results["off"], results["on"]

    def row(name: str, key: str) -> None:
        before, after = off[key], on[key]
        delta = 100 * (after / before - 1) if before else 0
        print(f"{name:30s} {before:10.3f} {after:10.3f} {delta:+9.2f}%")

    print(f"{'metric':30s} {'off':>10s} {'on':>10s} {'delta':>10s}")
    row("step wall (ms)", "wall_ms_per_step")
    row("GPU union busy (ms)", "gpu_busy_ms_per_step")
    row("GPU idle (ms)", "gpu_idle_ms_per_step")
    row("graph launches/step", "graph_launches_per_step")
    row("eager launches/step", "kernel_launches_per_step")
    row("target graph wall (ms)", "target_graph_ms_per_step")
    row("target graph busy (ms)", "target_graph_busy_ms_per_step")

    labels = sorted(set(off["categories"]) | set(on["categories"]))
    print("\nKernel families: cumulative device time per rank-step")
    print(f"{'family':30s} {'off ms':>10s} {'on ms':>10s} {'delta':>10s}")
    for label in labels:
        before = off["categories"].get(label, {}).get(
            "cumulative_ms_per_step", 0
        )
        after = on["categories"].get(label, {}).get("cumulative_ms_per_step", 0)
        delta = 100 * (after / before - 1) if before else 0
        print(f"{label:30s} {before:10.3f} {after:10.3f} {delta:+9.2f}%")

    if "tiered_layers" in off and "tiered_layers" in on:
        print("\nTiered routed layers")
        for key in (
            "layers_per_step",
            "hot_ms_per_step",
            "cold_ms_per_step",
            "marlin_sum_ms_per_step",
            "layer_span_ms_per_step",
            "timeline_overlap_percent",
        ):
            print(f"{key:30s} {off['tiered_layers'][key]:10.3f} "
                  f"{on['tiered_layers'][key]:10.3f}")

    print("\nLargest post-fix idle successors")
    for label, duration in list(
        on["idle_by_successor_ms_per_step"].items()
    )[:12]:
        print(f"{label:30s} {duration:10.3f} ms/step")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    results = {
        label: summarize_arm(args.trace_root / label) for label in ("off", "on")
    }
    print_report(results)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
