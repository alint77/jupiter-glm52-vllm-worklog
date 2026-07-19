#!/usr/bin/env python3
"""Analyze a four-rank DCP decode trace and build an analytical roofline."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

HIDDEN = 6144
LAYERS = 78
ROUTED_LAYERS = 75
INDEX_LAYERS = 21
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_CACHE_STRIDE = 132
ATTN_HEADS_PER_TP = 16
ATTN_QK_DIM = 576
ATTN_V_DIM = 512
ATTN_TOPK = 2048
MAIN_CACHE_STRIDE = 656
VOCAB_LOCAL = 38_720
EXPERT_BYTES_W4 = 19_464_192
EXPERT_FLOPS = 3 * 2 * HIDDEN * 2048

HBM_BYTES_S = 3.5e12
C2C_BYTES_S = 421e9
NVLINK_BYTES_S = 150e9
BF16_FLOPS_S = 630e12
FP8_FLOPS_S = 1_260e12


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def load_traces(trace_dir: Path) -> list[dict]:
    traces = []
    for path in sorted(trace_dir.glob("*.trace.json.gz")):
        with gzip.open(path, "rt") as handle:
            trace = json.load(handle)
        annotate_allgathers(trace)
        trace["_path"] = path.name
        traces.append(trace)
    if len(traces) != 4:
        raise ValueError(f"Expected four traces, found {len(traces)}")
    return sorted(traces, key=lambda trace: trace["distributedInfo"]["rank"])


def annotate_allgathers(trace: dict) -> None:
    events = kernels(trace)
    allgathers = [
        event for event in events if "ncclDevKernel_AllGather" in event["name"]
    ]
    for event in allgathers:
        grid = tuple(event.get("args", {}).get("grid", ()))
        if grid[:1] == (1,):
            event["_semantic_role"] = "DCP LSE all-gather"

    def claim(producer_predicate, semantic_role: str) -> None:
        producers = sorted(
            (event for event in events if producer_predicate(event)),
            key=lambda event: event["ts"],
        )
        for producer in producers:
            correlation = producer.get("args", {}).get("correlation")
            candidates = [
                event
                for event in allgathers
                if "_semantic_role" not in event
                and event.get("args", {}).get("correlation") == correlation
            ]
            if not candidates:
                candidates = [
                    event for event in allgathers if "_semantic_role" not in event
                ]
            if candidates:
                min(candidates, key=lambda event: abs(event["ts"] - producer["ts"]))[
                    "_semantic_role"
                ] = semantic_role

    claim(
        lambda event: "_pack_dcp_topk_candidates" in event["name"],
        "DCP candidate all-gather",
    )
    claim(
        lambda event: "nvjet_sm90_tst_192" in event["name"]
        and "splitK" not in event["name"],
        "Vocabulary all-gather",
    )
    for event in allgathers:
        event.setdefault("_semantic_role", "DCP query all-gather")


def kernels(trace: dict) -> list[dict]:
    return [
        event
        for event in trace["traceEvents"]
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]


def windows(trace: dict) -> list[dict]:
    annotations = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "gpu_user_annotation"
        and event.get("name", "").startswith("execute_context")
    ]
    by_tid = collections.Counter(event["tid"] for event in annotations)
    main_tid = by_tid.most_common(1)[0][0]
    return sorted(
        (event for event in annotations if event["tid"] == main_tid),
        key=lambda event: event["ts"],
    )


def trace_geometry(traces: list[dict]) -> tuple[int, int, int]:
    match = re.search(r"generation_(\d+)\((\d+)\)", windows(traces[0])[0]["name"])
    if match is None:
        raise ValueError("Could not infer generation geometry from trace annotation")
    sequences, target_tokens = map(int, match.groups())
    cycle_count = len(windows(traces[0])) - 2
    return sequences, target_tokens, cycle_count


def steady_events(trace: dict) -> tuple[list[dict], int]:
    step_windows = windows(trace)
    lo = step_windows[1]["ts"]
    hi = step_windows[-1]["ts"]
    return [event for event in kernels(trace) if lo <= event["ts"] < hi], len(
        step_windows
    ) - 2


def role(event: dict, allgather_roles: dict[tuple[int, ...], str]) -> str:
    category = event["cat"]
    name = event["name"]
    if category == "gpu_memcpy":
        return "Explicit copies"
    if category == "gpu_memset":
        return "Memsets"
    if "ncclDevKernel_AllGather" in name:
        if "_semantic_role" in event:
            return event["_semantic_role"]
        grid = tuple(event.get("args", {}).get("grid", ()))
        return allgather_roles.get(grid, "Other NCCL all-gather")
    if "ncclDevKernel_ReduceScatter" in name:
        return "DCP reduce-scatter"
    if "cross_device_reduce" in name:
        return "TP custom all-reduce"
    if "marlin_moe_wna16::Marlin" in name:
        return "Target routed W4 MoE"
    if name.startswith("void marlin::Marlin"):
        return "Target QKV-A W4"
    if "cutlass::device_kernel" in name:
        return "Target W4 Machete"
    if "paged_mqa_logits" in name:
        return "DSA full-context scan"
    if "cooperative_topk" in name or "StableTopKFromGathered" in name:
        return "DSA top-k"
    if "flash_fwd_splitkv" in name or "flash_fwd_mla_combine" in name:
        return "Sparse MLA"
    if "_correct_attn_cp_out_kernel" in name:
        return "DCP LSE correction"
    if "dcp_topk" in name or "convert_req_index_to_global" in name:
        return "DCP/index metadata"
    if "linearcute_dsl_ll_bf16_" in name:
        return "Target MoE router"
    if "nvjet_sm90_tst_192" in name and "splitK" in name:
        return "MTP BF16 projection"
    if "nvjet_sm90_tst_192" in name:
        return "Vocabulary GEMM"
    if "nvjet_sm90" in name:
        return "MLA/DSA BF16 GEMMs"
    if "deep_gemm::fp8_gemm" in name or "tensorrt_llm::kernels" in name:
        return "MTP FP8 block"
    if "triton_tem_fused_mm_t_3" in name:
        return "MTP BF16 projection"
    if any(
        token in name
        for token in (
            "moe_sum_vec_kernel",
            "act_and_mul_kernel",
            "moe_align_block_size",
            "count_and_sort_expert",
            "grouped_topk_fused_small_expert",
        )
    ):
        return "Routed MoE support"
    if any(
        token in name
        for token in (
            "concat_and_cache_ds_mla",
            "indexer_k_quant_and_cache",
            "fused_indexer_q_rope_quant",
            "get_mla_metadata",
        )
    ):
        return "Attention/index support"
    return "Elementwise and metadata"


def kernel_label(event: dict) -> str:
    name = event["name"].removeprefix("void ")
    aliases = (
        ("marlin_moe_wna16::Marlin", "marlin_moe_wna16::Marlin"),
        ("cutlass::device_kernel", "Machete GEMM"),
        ("linearcute_dsl_ll_bf16_dotprod", "BF16 router GEMM"),
        ("StableTopKFromGathered", "DCP stable top-k"),
        ("ncclDevKernel_AllGather", "NCCL AllGather"),
        ("ncclDevKernel_ReduceScatter", "NCCL ReduceScatter"),
        ("cross_device_reduce", "vLLM custom all-reduce"),
    )
    for token, label in aliases:
        if token in name:
            return label
    return re.split(r"[<(]", name, maxsplit=1)[0]


def union_time(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged = [list(interval) for interval in sorted(intervals)]
    out = [merged[0]]
    for start, end in merged[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return sum(end - start for start, end in out)


def timeline(traces: list[dict], allgather_roles: dict) -> dict:
    rank_rows = []
    role_solo = collections.defaultdict(list)
    depth_totals = collections.Counter()
    gap_values = []
    for trace in traces:
        step_windows = windows(trace)
        cycles = []
        rank_solo = collections.Counter()
        for index in range(1, len(step_windows) - 1):
            lo = step_windows[index]["ts"]
            hi = step_windows[index + 1]["ts"]
            step_kernels = [event for event in kernels(trace) if lo <= event["ts"] < hi]
            intervals = [
                (event["ts"], min(event["ts"] + event["dur"], hi))
                for event in step_kernels
            ]
            points = []
            for event, (start, end) in zip(step_kernels, intervals, strict=True):
                component = role(event, allgather_roles)
                points.append((start, 1, component))
                points.append((end, -1, component))
            points.sort(key=lambda point: (point[0], point[1]))
            active = collections.Counter()
            previous = lo
            idle = 0.0
            for timestamp, delta, component in points:
                duration = timestamp - previous
                if not active:
                    idle += duration
                    if duration > 0:
                        gap_values.append(duration)
                else:
                    depth_totals[min(sum(active.values()), 4)] += duration
                    if len(active) == 1:
                        rank_solo[next(iter(active))] += duration
                active[component] += delta
                if active[component] == 0:
                    del active[component]
                previous = timestamp
            if previous < hi:
                idle += hi - previous
                gap_values.append(hi - previous)
            cycles.append(
                {
                    "wall_us": hi - lo,
                    "target_us": step_windows[index]["dur"],
                    "draft_tail_us": hi - lo - step_windows[index]["dur"],
                    "union_busy_us": union_time(intervals),
                    "cumulative_activity_us": sum(
                        event["dur"] for event in step_kernels
                    ),
                    "idle_us": idle,
                    "kernel_count": len(step_kernels),
                }
            )
        rank = trace["distributedInfo"]["rank"]
        rank_rows.append(
            {
                "rank": rank,
                **{
                    key: statistics.mean(cycle[key] for cycle in cycles)
                    for key in cycles[0]
                },
            }
        )
        for component, duration in rank_solo.items():
            role_solo[component].append(duration / len(cycles))
    total_depth = sum(depth_totals.values())
    return {
        "per_rank": rank_rows,
        "mean": {
            key: statistics.mean(row[key] for row in rank_rows)
            for key in rank_rows[0]
            if key != "rank"
        },
        "busy_depth_pct": {
            str(depth): 100 * duration / total_depth
            for depth, duration in sorted(depth_totals.items())
        },
        "solo_us_per_step": {
            component: statistics.mean(values)
            for component, values in sorted(
                role_solo.items(), key=lambda item: -statistics.mean(item[1])
            )
        },
        "idle_gap_us": {
            "mean": statistics.mean(gap_values),
            "p50": percentile(gap_values, 0.5),
            "p90": percentile(gap_values, 0.9),
            "p99": percentile(gap_values, 0.99),
            "max": max(gap_values),
        },
    }


def inventory(traces: list[dict], allgather_roles: dict) -> tuple[list[dict], dict]:
    rank_groups = []
    rank_roles = []
    for trace in traces:
        events, cycle_count = steady_events(trace)
        groups = collections.defaultdict(list)
        roles = collections.Counter()
        for event in events:
            component = role(event, allgather_roles)
            args = event.get("args", {})
            key = (
                component,
                kernel_label(event),
                tuple(args.get("grid", ())),
                tuple(args.get("block", ())),
            )
            groups[key].append(event)
            roles[component] += event["dur"] / cycle_count
        rank_groups.append((groups, cycle_count))
        rank_roles.append(roles)

    rows = []
    keys = set().union(*(groups for groups, _ in rank_groups))
    for component, label, grid, block in keys:
        times = []
        calls = []
        durations = []
        occupancies = []
        for groups, cycle_count in rank_groups:
            events = groups.get((component, label, grid, block), [])
            times.append(sum(event["dur"] for event in events) / cycle_count)
            calls.append(len(events) / cycle_count)
            durations.extend(event["dur"] for event in events)
            occupancies.extend(
                event.get("args", {}).get("est. achieved occupancy %")
                for event in events
                if "est. achieved occupancy %" in event.get("args", {})
            )
        mean_calls = statistics.mean(calls)
        if mean_calls == 0:
            continue
        rows.append(
            {
                "role": component,
                "kernel": label,
                "grid": "x".join(map(str, grid)),
                "block": "x".join(map(str, block)),
                "calls_per_step": mean_calls,
                "time_us_per_step": statistics.mean(times),
                "rank_min_us": min(times),
                "rank_max_us": max(times),
                "mean_call_us": statistics.mean(durations),
                "p50_call_us": percentile(durations, 0.5),
                "p90_call_us": percentile(durations, 0.9),
                "p99_call_us": percentile(durations, 0.99),
                "launch_occupancy_pct": (
                    statistics.mean(occupancies) if occupancies else None
                ),
            }
        )
    rows.sort(key=lambda row: row["time_us_per_step"], reverse=True)
    role_times = {
        component: statistics.mean(rank[component] for rank in rank_roles)
        for component in set().union(*rank_roles)
    }
    return rows, dict(sorted(role_times.items(), key=lambda item: -item[1]))


def matching_time(traces: list[dict], predicate) -> float:
    values = []
    for trace in traces:
        events, cycles = steady_events(trace)
        values.append(
            sum(event["dur"] for event in events if predicate(event)) / cycles
        )
    return statistics.mean(values)


def predecessor_time(traces: list[dict]) -> float:
    values = []
    for trace in traces:
        events, cycles = steady_events(trace)
        streams = collections.defaultdict(list)
        for event in events:
            if event["cat"] == "kernel":
                streams[event.get("tid")].append(event)
        total = 0.0
        for stream_events in streams.values():
            stream_events.sort(key=lambda event: event["ts"])
            for index, event in enumerate(stream_events):
                if "_fused_indexer_q_rope_quant_kernel" in event["name"] and index:
                    previous = stream_events[index - 1]
                    if "deep_gemm::fp8_gemm" not in previous["name"]:
                        total += previous["dur"]
        values.append(total / cycles)
    return statistics.mean(values)


def roofline_row(
    name: str,
    time_us: float,
    flops: float,
    byte_count: float,
    compute_ceiling: float = BF16_FLOPS_S,
    bandwidth_ceiling: float = HBM_BYTES_S,
    note: str = "",
) -> dict:
    seconds = time_us * 1e-6
    intensity = flops / byte_count
    achieved = flops / seconds
    roof = min(compute_ceiling, intensity * bandwidth_ceiling)
    return {
        "name": name,
        "time_us_per_step": time_us,
        "flops_per_step": flops,
        "logical_bytes_per_step": byte_count,
        "arithmetic_intensity_flop_per_byte": intensity,
        "achieved_tflops": achieved / 1e12,
        "effective_bandwidth_gbs": byte_count / seconds / 1e9,
        "roof_tflops": roof / 1e12,
        "roof_efficiency_pct": 100 * achieved / roof,
        "note": note,
    }


def byte_row(name: str, time_us: float, byte_count: float, note: str) -> dict:
    bandwidth = byte_count / (time_us * 1e-6)
    return {
        "name": name,
        "time_us_per_step": time_us,
        "flops_per_step": None,
        "logical_bytes_per_step": byte_count,
        "arithmetic_intensity_flop_per_byte": None,
        "achieved_tflops": None,
        "effective_bandwidth_gbs": bandwidth / 1e9,
        "roof_tflops": None,
        "roof_efficiency_pct": 100 * bandwidth / HBM_BYTES_S,
        "note": note,
    }


def modeled_roofline(
    traces: list[dict], sequences: int, target_q: int, context: int, dcp_size: int
) -> list[dict]:
    draft_q = sequences
    w4_bytes = 0.5 + 2 / 128
    local_context = context / dcp_size
    local_topk = ATTN_TOPK / dcp_size
    mla_heads = ATTN_HEADS_PER_TP * dcp_size

    qkv_weights = HIDDEN * 2624 * LAYERS
    machete_weights = (
        (2048 * 4096 + 4096 * HIDDEN) * LAYERS
        + (HIDDEN * 1024 + 512 * HIDDEN) * ROUTED_LAYERS
        + (HIDDEN * 6144 + 3072 * HIDDEN) * 3
    )
    contractions_weights = (
        ATTN_HEADS_PER_TP * ATTN_QK_DIM * 192 + ATTN_HEADS_PER_TP * ATTN_V_DIM * 256
    )
    mla_queries = target_q * LAYERS + 3 * draft_q
    index_queries = target_q * INDEX_LAYERS + 3 * draft_q
    mtp_shapes = [
        (2624, HIDDEN),
        (4096, 2048),
        (4096, 2048),
        (HIDDEN, 4096),
        (1024, HIDDEN),
        (HIDDEN, 512),
    ]
    mtp_weights = sum(m * n for m, n in mtp_shapes)

    is_name = lambda token: lambda event: token in event["name"]
    rows = [
        roofline_row(
            "Target QKV-A W4 Marlin",
            matching_time(
                traces,
                lambda event: event["name"].startswith("void marlin::Marlin"),
            ),
            2 * target_q * qkv_weights,
            qkv_weights * w4_bytes,
        ),
        roofline_row(
            "Target W4 Machete projections/MLPs",
            matching_time(traces, is_name("cutlass::device_kernel")),
            2 * target_q * machete_weights,
            machete_weights * w4_bytes,
        ),
        roofline_row(
            "Target MoE router BF16",
            matching_time(traces, is_name("linearcute_dsl_ll_bf16_")),
            2 * target_q * HIDDEN * 256 * ROUTED_LAYERS,
            HIDDEN * 256 * 2 * ROUTED_LAYERS,
        ),
        roofline_row(
            "MLA W_UK/W_UV BF16 contractions",
            matching_time(
                traces,
                lambda event: "nvjet_sm90_tst_64x" in event["name"]
                and ("_bz_NNT" in event["name"] or "_2x1_v_bz_TNT" in event["name"]),
            ),
            2 * contractions_weights * mla_queries,
            contractions_weights * 2 * (LAYERS + 3),
        ),
        roofline_row(
            "Target DSA Wq BF16",
            predecessor_time(traces),
            2 * target_q * 2048 * 4096 * INDEX_LAYERS,
            2048 * 4096 * 2 * INDEX_LAYERS,
        ),
        roofline_row(
            "DSA WK+score projection BF16",
            matching_time(traces, is_name("nvjet_sm90_tss")),
            2 * HIDDEN * 160 * index_queries,
            HIDDEN * 160 * 2 * (INDEX_LAYERS + 3),
        ),
        roofline_row(
            "DSA local-shard FP8 scan",
            matching_time(traces, is_name("paged_mqa_logits")),
            2 * local_context * INDEX_DIM * INDEX_HEADS * index_queries,
            local_context * INDEX_CACHE_STRIDE * (INDEX_LAYERS + 3),
            compute_ceiling=FP8_FLOPS_S,
            note="Each rank scans one quarter of the 400K index cache.",
        ),
        byte_row(
            "DSA local + global stable top-k",
            matching_time(
                traces,
                lambda event: "cooperative_topk" in event["name"]
                or "StableTopKFromGathered" in event["name"],
            ),
            local_context * 4 * index_queries
            + index_queries * dcp_size * ATTN_TOPK * 2 * 4,
            "Minimum local-logit and gathered-candidate reads; selection FLOPs "
            "are not modeled.",
        ),
        roofline_row(
            "Sparse MLA FP8 split+combine",
            matching_time(
                traces,
                lambda event: "flash_fwd_splitkv" in event["name"]
                or "flash_fwd_mla_combine" in event["name"],
            ),
            2 * mla_heads * local_topk * (ATTN_QK_DIM + ATTN_V_DIM) * mla_queries,
            MAIN_CACHE_STRIDE * local_topk * mla_queries,
            note=(
                "Useful work for roughly 512 rank-owned candidates. DCP compacts "
                "them into a fixed 2,048-entry row whose -1 tail the kernel skips; "
                "64 gathered heads are real, with no 16-to-64 head padding."
            ),
        ),
        roofline_row(
            "MTP eh_proj BF16",
            matching_time(
                traces,
                lambda event: (
                    "nvjet_sm90_tst_192" in event["name"] and "splitK" in event["name"]
                )
                or "triton_tem_fused_mm_t_3" in event["name"],
            ),
            3 * 2 * draft_q * 12_288 * HIDDEN,
            3 * 12_288 * HIDDEN * 2,
        ),
        roofline_row(
            "MTP FP8 dense/index/shared linears",
            matching_time(
                traces,
                lambda event: "deep_gemm::fp8_gemm_kernel_swapAB" in event["name"]
                and "GroupedWithOffset" not in event["name"],
            ),
            3 * 2 * draft_q * mtp_weights,
            3 * mtp_weights,
            compute_ceiling=FP8_FLOPS_S,
        ),
        roofline_row(
            "MTP FP8 routed experts",
            matching_time(
                traces,
                lambda event: "deep_gemm::fp8_gemm_kernel_swapAB" in event["name"]
                and "GroupedWithOffset" in event["name"],
            ),
            3 * 2 * draft_q * EXPERT_FLOPS,
            3 * 2 * draft_q * 3 * HIDDEN * 2048,
            compute_ceiling=FP8_FLOPS_S,
            note="Assumes two expert assignments per rank/token/pass.",
        ),
        roofline_row(
            "BF16 vocabulary projection",
            matching_time(
                traces,
                lambda event: "nvjet_sm90_tst_192" in event["name"]
                and "splitK" not in event["name"],
            ),
            2 * (target_q + 3 * draft_q) * HIDDEN * VOCAB_LOCAL,
            4 * HIDDEN * VOCAB_LOCAL * 2,
        ),
    ]
    return rows


def routed_bounds(traces: list[dict], target_q: int) -> dict:
    rank_stats = []
    for trace in traces:
        events, cycles = steady_events(trace)
        step_windows = windows(trace)
        totals = collections.Counter()
        for index in range(1, len(step_windows) - 1):
            lo, hi = step_windows[index]["ts"], step_windows[index + 1]["ts"]
            routed = sorted(
                (
                    event
                    for event in events
                    if lo <= event["ts"] < hi
                    and "marlin_moe_wna16::Marlin" in event["name"]
                ),
                key=lambda event: event["ts"],
            )
            clusters = []
            for event in routed:
                end = event["ts"] + event["dur"]
                if not clusters or event["ts"] - clusters[-1]["end"] > 50:
                    clusters.append({"events": [event], "end": end})
                else:
                    clusters[-1]["events"].append(event)
                    clusters[-1]["end"] = max(clusters[-1]["end"], end)
            for cluster in clusters:
                by_stream = collections.defaultdict(list)
                for event in cluster["events"]:
                    by_stream[event.get("tid")].append(event)
                ordered = sorted(
                    by_stream.values(),
                    key=lambda group: min(event["ts"] for event in group),
                )
                if len(ordered) == 1:
                    hot, cold = ordered[0], []
                else:
                    cold = ordered[0]
                    hot = [event for group in ordered[1:] for event in group]
                hot_time = sum(event["dur"] for event in hot)
                cold_time = sum(event["dur"] for event in cold)
                totals["hot_us"] += hot_time
                totals["cold_us"] += cold_time
                totals["serial_us"] += hot_time + cold_time
                totals["union_us"] += union_time(
                    [
                        (event["ts"], event["ts"] + event["dur"])
                        for event in cluster["events"]
                    ]
                )
                totals["cold_active"] += bool(cold)
                totals["clusters"] += 1
        rank_stats.append(
            {
                "rank": trace["distributedInfo"]["rank"],
                **{key: value / cycles for key, value in totals.items()},
            }
        )
    hot_us = statistics.mean(row["hot_us"] for row in rank_stats)
    cold_us = statistics.mean(row["cold_us"] for row in rank_stats)
    cold_min = statistics.mean(row["cold_active"] for row in rank_stats)
    assignments = ROUTED_LAYERS * target_q * 8 / 4
    cold_max = min(assignments, cold_us * 1e-6 * C2C_BYTES_S / EXPERT_BYTES_W4)
    hot_min, hot_max = assignments - cold_max, assignments - cold_min

    def tier(low: float, high: float, time_us: float, ceiling: float) -> dict:
        return {
            "assignment_bound": [low, high],
            "logical_bytes_bound": [low * EXPERT_BYTES_W4, high * EXPERT_BYTES_W4],
            "effective_bandwidth_gbs": [
                low * EXPERT_BYTES_W4 / (time_us * 1e-6) / 1e9,
                high * EXPERT_BYTES_W4 / (time_us * 1e-6) / 1e9,
            ],
            "roof_efficiency_pct": [
                100 * low * EXPERT_BYTES_W4 / (time_us * 1e-6) / ceiling,
                100 * high * EXPERT_BYTES_W4 / (time_us * 1e-6) / ceiling,
            ],
            "time_us_per_step": time_us,
        }

    serial = statistics.mean(row["serial_us"] for row in rank_stats)
    observed = statistics.mean(row["union_us"] for row in rank_stats)
    return {
        "total_assignments_per_rank_step": assignments,
        "rank_stats": rank_stats,
        "observed_overlap": {
            "serial_kernel_us": serial,
            "union_kernel_us": observed,
            "saved_us": serial - observed,
            "saved_pct": 100 * (serial - observed) / serial,
        },
        "hot_hbm": tier(hot_min, hot_max, hot_us, HBM_BYTES_S),
        "cold_c2c": tier(cold_min, cold_max, cold_us, C2C_BYTES_S),
    }


def communication(
    role_times: dict, sequences: int, target_q: int, dcp_size: int
) -> list[dict]:
    draft_q = sequences
    mla_queries = target_q * LAYERS + 3 * draft_q
    index_queries = target_q * INDEX_LAYERS + 3 * draft_q
    local_heads = ATTN_HEADS_PER_TP
    remote_factor = dcp_size - 1
    payloads = {
        "DCP query all-gather": remote_factor * mla_queries * local_heads * ATTN_QK_DIM,
        "DCP LSE all-gather": remote_factor * mla_queries * local_heads * dcp_size * 4,
        "DCP candidate all-gather": remote_factor * index_queries * ATTN_TOPK * 2 * 4,
        "DCP reduce-scatter": remote_factor
        * mla_queries
        * local_heads
        * ATTN_V_DIM
        * 2,
        "TP custom all-reduce": remote_factor
        * (157 * target_q * HIDDEN * 2 + 9 * draft_q * HIDDEN * 2),
        "Vocabulary all-gather": remote_factor
        * (target_q + 3 * draft_q)
        * VOCAB_LOCAL
        * 2,
    }
    rows = []
    for name, byte_count in payloads.items():
        time_us = role_times.get(name, 0.0)
        rows.append(
            {
                "name": name,
                "time_us_per_step": time_us,
                "logical_remote_bytes_per_rank_step": byte_count,
                "effective_gbs": byte_count / (time_us * 1e-6) / 1e9,
                "link_efficiency_pct": 100
                * byte_count
                / (time_us * 1e-6)
                / NVLINK_BYTES_S,
            }
        )
    return rows


def write_inventory(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_roofline(path: Path, rows: list[dict]) -> None:
    points = [row for row in rows if row["achieved_tflops"] is not None]
    x_values = [10 ** (value / 60) for value in range(-60, 190)]
    figure, (roof_axis, efficiency_axis) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    roof_axis.loglog(
        x_values, [min(630, 3.5 * value) for value in x_values], label="BF16/HBM"
    )
    roof_axis.loglog(
        x_values,
        [min(1260, 3.5 * value) for value in x_values],
        "--",
        label="FP8/HBM",
    )
    for row in points:
        x = row["arithmetic_intensity_flop_per_byte"]
        y = row["achieved_tflops"]
        roof_axis.scatter(x, y)
        roof_axis.annotate(" ".join(row["name"].split()[:3]), (x, y), fontsize=7)
    roof_axis.set_xlabel("Arithmetic intensity (FLOP/byte)")
    roof_axis.set_ylabel("Achieved TFLOP/s")
    roof_axis.grid(True, which="both", alpha=0.2)
    roof_axis.legend()

    names = [row["name"] for row in rows]
    values = [row["roof_efficiency_pct"] for row in rows]
    efficiency_axis.barh(names, values)
    efficiency_axis.invert_yaxis()
    efficiency_axis.set_xlabel("Analytical roof efficiency (%)")
    efficiency_axis.grid(True, axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".svg"))
    figure.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("prefix")
    parser.add_argument("--context", type=int, default=399_744)
    parser.add_argument("--dcp-size", type=int, default=4)
    args = parser.parse_args()

    traces = load_traces(args.trace_dir)
    sequences, target_q, cycles = trace_geometry(traces)
    allgather_roles = {}
    kernel_rows, role_times = inventory(traces, allgather_roles)
    roofline = modeled_roofline(
        traces, sequences, target_q, args.context, args.dcp_size
    )
    summary = {
        "method": {
            "trace_directory": str(args.trace_dir),
            "trace_files": [trace["_path"] for trace in traces],
            "steady_cycles": cycles,
            "sequences": sequences,
            "target_verification_tokens": target_q,
            "draft_tokens_per_pass": sequences,
            "context_tokens": args.context,
            "dcp_size": args.dcp_size,
            "trace_has_hardware_counters": False,
            "duration_source": "PyTorch profiler CUDA events",
            "traffic_source": "Model geometry and kernel contracts",
            "allgather_attribution": (
                "LSE by launch shape; candidate and vocabulary collectives are "
                "matched one-to-one with their producer kernels; the remainder "
                "are query all-gathers. This handles shared NCCL launch shapes."
            ),
        },
        "ceilings": {
            "hbm_bytes_per_second": HBM_BYTES_S,
            "grace_c2c_bytes_per_second": C2C_BYTES_S,
            "peer_nvlink_bytes_per_second": NVLINK_BYTES_S,
            "bf16_flops_per_second": BF16_FLOPS_S,
            "fp8_flops_per_second": FP8_FLOPS_S,
        },
        "timeline": timeline(traces, allgather_roles),
        "activity_by_role_us_per_step": role_times,
        "roofline": roofline,
        "routed_moe_bounds": routed_bounds(traces, target_q),
        "communication": communication(role_times, sequences, target_q, args.dcp_size),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.prefix
    write_inventory(stem.with_name(stem.name + "-kernels.csv"), kernel_rows)
    with stem.with_name(stem.name + "-summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    plot_roofline(stem.with_name(stem.name + "-roofline"), roofline)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
