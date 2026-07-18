#!/usr/bin/env python3
"""Build an analytical roofline from the four-rank PyTorch traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

STEPS = 8
HIDDEN = 6144
LAYERS = 78
ROUTED_LAYERS = 75
CONTEXT = 399_744
INDEX_HEADS = 32
INDEX_DIM = 128
ATTN_HEADS = 16
PADDED_ATTN_HEADS = 64
ATTN_TOPK = 2048
MLA_QK_DIM = 576
MLA_V_DIM = 512
MAIN_CACHE_STRIDE = 656
INDEX_CACHE_STRIDE = 132
VOCAB_LOCAL = 38_720
EXPERT_BYTES_W4 = 19_464_192
EXPERT_FLOPS = 3 * 2 * HIDDEN * 2048

HBM_BYTES_S = 3.5e12
C2C_BYTES_S = 421e9
NVLINK_BYTES_S = 150e9
BF16_FLOPS_S = 630e12
FP8_FLOPS_S = 1_260e12


def short_name(name: str) -> str:
    name = name.removeprefix("void ")
    for separator in ("<", "("):
        name = name.split(separator, 1)[0]
    return name


def role(name: str, category: str) -> str:
    if category == "gpu_memcpy":
        return "Explicit copies"
    if category == "gpu_memset":
        return "Memsets"
    if "marlin_moe_wna16" in name:
        return "Routed W4 MoE"
    if "cross_device_reduce" in name:
        return "Custom all-reduce"
    if "cutlass::device_kernel" in name or name.startswith("marlin::Marlin"):
        return "Target W4 linear"
    if "flash_fwd_splitkv" in name or "flash_fwd_mla_combine" in name:
        return "Sparse MLA"
    if any(
        token in name
        for token in (
            "concat_and_cache_ds_mla",
            "ConcatMLAQKernel",
            "convert_req_index_to_global",
            "get_mla_metadata",
        )
    ):
        return "Sparse MLA support"
    if any(
        token in name
        for token in (
            "paged_mqa_logits",
            "cooperative_topk",
            "fused_indexer_q_rope_quant",
            "indexer_k_quant_and_cache",
            "triton_tem_fused_mm_t_7",
            "triton_tem_fused_mm_t_6",
            "nvjet_sm90_tss",
            "nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT",
        )
    ):
        return "DSA indexer"
    if any(
        token in name
        for token in (
            "deep_gemm::fp8_gemm",
            "tensorrt_llm::kernels",
            "router_gemm_kernel",
        )
    ):
        return "MTP FP8 block"
    if "triton_tem_fused_mm_t_3" in name:
        return "MTP BF16 projection"
    if "nvjet_sm90_tst_192" in name or "ncclDevKernel_AllGather" in name:
        return "Vocabulary output"
    if "nvjet_sm90_tst_64" in name:
        return "MLA BF16 contractions"
    if "kernel_cutlass_kernel_vllmmodel" in name:
        return "MoE router"
    if any(
        token in name
        for token in (
            "vllm::moe::",
            "vllm::act_and_mul",
            "grouped_topk_fused_small_expert",
        )
    ):
        return "Routed MoE support"
    return "Elementwise and metadata"


def load_traces(trace_dir: Path) -> list[dict]:
    traces = []
    for path in sorted(trace_dir.glob("*.trace.json.gz")):
        with gzip.open(path, "rt") as handle:
            traces.append(json.load(handle))
    if len(traces) != 4:
        raise ValueError(f"Expected four traces, found {len(traces)}")
    return sorted(traces, key=lambda trace: trace["distributedInfo"]["rank"])


def kernel_events(trace: dict) -> list[dict]:
    return [
        event
        for event in trace["traceEvents"]
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]


def matching_time_us(traces: list[dict], predicate) -> float:
    rank_times = []
    for trace in traces:
        total = sum(event["dur"] for event in kernel_events(trace) if predicate(event))
        rank_times.append(total / STEPS)
    return statistics.mean(rank_times)


def matching_calls(traces: list[dict], predicate) -> float:
    return statistics.mean(
        sum(predicate(event) for event in kernel_events(trace)) / STEPS
        for trace in traces
    )


def predecessor_time_us(traces: list[dict], anchor: str, exclude: str) -> float:
    rank_times = []
    for trace in traces:
        streams = defaultdict(list)
        for event in kernel_events(trace):
            if event["cat"] == "kernel":
                streams[event.get("tid")].append(event)
        total = 0.0
        for events in streams.values():
            events.sort(key=lambda event: event["ts"])
            for index, event in enumerate(events):
                if anchor in event["name"] and exclude not in events[index - 1]["name"]:
                    total += events[index - 1]["dur"]
        rank_times.append(total / STEPS)
    return statistics.mean(rank_times)


def contains(*tokens: str):
    return lambda event: any(token in event["name"] for token in tokens)


def inventory(traces: list[dict]) -> tuple[list[dict], dict[str, float]]:
    per_rank = []
    for trace in traces:
        grouped = defaultdict(
            lambda: {
                "calls": 0,
                "time_us": 0.0,
                "bytes": 0,
                "occupancy": [],
                "grids": set(),
                "blocks": set(),
                "streams": set(),
            }
        )
        for event in kernel_events(trace):
            category = event["cat"]
            name = short_name(event["name"])
            item = grouped[(category, name)]
            item["calls"] += 1
            item["time_us"] += event["dur"]
            args = event.get("args", {})
            item["bytes"] += args.get("bytes", 0)
            if "est. achieved occupancy %" in args:
                item["occupancy"].append(args["est. achieved occupancy %"])
            if args.get("grid"):
                item["grids"].add(tuple(args["grid"]))
            if args.get("block"):
                item["blocks"].add(tuple(args["block"]))
            item["streams"].add(event.get("tid"))
        per_rank.append(grouped)

    keys = sorted(set().union(*(rank.keys() for rank in per_rank)))
    rows = []
    role_times = defaultdict(float)
    for category, name in keys:
        entries = [rank[(category, name)] for rank in per_rank]
        times = [entry["time_us"] / STEPS for entry in entries]
        calls = [entry["calls"] / STEPS for entry in entries]
        occupancies = [value for entry in entries for value in entry["occupancy"]]
        group_role = role(name, category)
        mean_time = statistics.mean(times)
        role_times[group_role] += mean_time
        rows.append(
            {
                "role": group_role,
                "category": category,
                "kernel": name,
                "calls_per_step": statistics.mean(calls),
                "time_us_per_step": mean_time,
                "rank_min_us": min(times),
                "rank_max_us": max(times),
                "mean_call_us": mean_time / statistics.mean(calls),
                "logical_bytes_per_step": statistics.mean(
                    entry["bytes"] / STEPS for entry in entries
                ),
                "profiler_launch_occupancy_pct": (
                    statistics.mean(occupancies) if occupancies else None
                ),
                "grids": sorted(set().union(*(entry["grids"] for entry in entries))),
                "blocks": sorted(set().union(*(entry["blocks"] for entry in entries))),
                "streams": sorted(
                    set().union(*(entry["streams"] for entry in entries))
                ),
            }
        )
    rows.sort(key=lambda row: row["time_us_per_step"], reverse=True)
    return rows, dict(sorted(role_times.items(), key=lambda item: -item[1]))


def roofline_row(
    name: str,
    domain: str,
    calls: float,
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
    bandwidth = byte_count / seconds
    roof = min(compute_ceiling, intensity * bandwidth_ceiling)
    return {
        "name": name,
        "domain": domain,
        "calls_per_step": calls,
        "time_us_per_step": time_us,
        "flops_per_step": flops,
        "logical_bytes_per_step": byte_count,
        "arithmetic_intensity_flop_per_byte": intensity,
        "achieved_tflops": achieved / 1e12,
        "effective_bandwidth_gbs": bandwidth / 1e9,
        "roof_tflops": roof / 1e12,
        "roof_efficiency_pct": 100 * achieved / roof,
        "note": note,
    }


def byte_row(
    name: str,
    domain: str,
    calls: float,
    time_us: float,
    byte_count: float,
    bandwidth_ceiling: float,
    note: str = "",
) -> dict:
    bandwidth = byte_count / (time_us * 1e-6)
    return {
        "name": name,
        "domain": domain,
        "calls_per_step": calls,
        "time_us_per_step": time_us,
        "flops_per_step": None,
        "logical_bytes_per_step": byte_count,
        "arithmetic_intensity_flop_per_byte": None,
        "achieved_tflops": None,
        "effective_bandwidth_gbs": bandwidth / 1e9,
        "roof_tflops": None,
        "roof_efficiency_pct": 100 * bandwidth / bandwidth_ceiling,
        "note": note,
    }


def modeled_rows(traces: list[dict]) -> list[dict]:
    w4_bytes_per_weight = 0.5 + 2 / 128

    qkv_flops = 2 * 4 * HIDDEN * 2624 * LAYERS
    qkv_bytes = HIDDEN * 2624 * w4_bytes_per_weight * LAYERS
    qkv_time = matching_time_us(traces, contains("marlin::Marlin<"))

    attention_flops = 2 * 4 * (2048 * 4096 + 4096 * HIDDEN) * LAYERS
    sparse_shared_flops = 2 * 4 * (HIDDEN * 1024 + 512 * HIDDEN) * 75
    dense_mlp_flops = 2 * 4 * (HIDDEN * 6144 + 3072 * HIDDEN) * 3
    machete_flops = attention_flops + sparse_shared_flops + dense_mlp_flops
    attention_bytes = (2048 * 4096 + 4096 * HIDDEN) * w4_bytes_per_weight * LAYERS
    sparse_shared_bytes = (HIDDEN * 1024 + 512 * HIDDEN) * w4_bytes_per_weight * 75
    dense_mlp_bytes = (HIDDEN * 6144 + 3072 * HIDDEN) * w4_bytes_per_weight * 3
    machete_bytes = attention_bytes + sparse_shared_bytes + dense_mlp_bytes
    machete_time = matching_time_us(traces, contains("cutlass::device_kernel"))

    router_flops = 2 * 4 * HIDDEN * 256 * ROUTED_LAYERS
    router_bytes = HIDDEN * 256 * 2 * ROUTED_LAYERS
    router_time = matching_time_us(traces, contains("kernel_cutlass_kernel_vllmmodel"))

    contractions_flops = (
        2
        * (ATTN_HEADS * MLA_QK_DIM * 192 + ATTN_HEADS * MLA_V_DIM * 256)
        * (4 * LAYERS + 3)
    )
    contractions_bytes = (
        (ATTN_HEADS * MLA_QK_DIM * 192 + ATTN_HEADS * MLA_V_DIM * 256)
        * 2
        * (LAYERS + 3)
    )
    contractions_time = matching_time_us(
        traces,
        contains(
            "nvjet_sm90_tst_64x8_64x16_4x1_v_bz_NNT",
            "nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT",
        ),
    )

    wq_flops = 2 * 4 * 2048 * 4096 * 21
    wq_bytes = 2048 * 4096 * 2 * 21
    wq_time = predecessor_time_us(
        traces,
        "_fused_indexer_q_rope_quant_kernel",
        "deep_gemm::fp8_gemm_kernel_swapAB",
    )

    wk_flops = 2 * HIDDEN * 160 * (4 * 21 + 3)
    wk_bytes = HIDDEN * 160 * 2 * (21 + 3)
    wk_time = matching_time_us(traces, contains("nvjet_sm90_tss"))

    index_queries = 4 * 21 + 3
    index_flops = 2 * CONTEXT * INDEX_DIM * INDEX_HEADS * index_queries
    index_bytes = CONTEXT * INDEX_CACHE_STRIDE * (21 + 3)
    index_time = matching_time_us(
        traces, contains("deep_gemm::sm90_fp8_paged_mqa_logits")
    )

    topk_bytes = CONTEXT * 4 * index_queries
    topk_time = matching_time_us(traces, contains("cooperative_topk_cs8"))

    mla_queries = 4 * LAYERS + 3
    mla_flops = (
        2 * PADDED_ATTN_HEADS * ATTN_TOPK * (MLA_QK_DIM + MLA_V_DIM) * mla_queries
    )
    mla_bytes = MAIN_CACHE_STRIDE * ATTN_TOPK * mla_queries
    mla_time = matching_time_us(
        traces, contains("flash_fwd_splitkv", "flash_fwd_mla_combine")
    )

    eh_flops = 3 * 2 * 12_288 * HIDDEN
    eh_bytes = 3 * 12_288 * HIDDEN * 2
    eh_time = matching_time_us(traces, contains("triton_tem_fused_mm_t_3"))

    normal_fp8_shapes = [
        (2624, HIDDEN),
        (4096, 2048),
        (4096, 2048),
        (HIDDEN, 4096),
        (1024, HIDDEN),
        (HIDDEN, 512),
    ]
    mtp_dense_weights = sum(m * n for m, n in normal_fp8_shapes)
    mtp_dense_flops = 3 * 2 * mtp_dense_weights
    mtp_dense_bytes = 3 * mtp_dense_weights
    mtp_dense_time = matching_time_us(
        traces,
        lambda event: (
            "deep_gemm::fp8_gemm_kernel_swapAB" in event["name"]
            and "GroupedWithOffset" not in event["name"]
        ),
    )

    mtp_moe_flops = 3 * 2 * EXPERT_FLOPS
    mtp_moe_bytes = 3 * 2 * 3 * HIDDEN * 2048
    mtp_moe_time = matching_time_us(
        traces,
        lambda event: (
            "deep_gemm::fp8_gemm_kernel_swapAB" in event["name"]
            and "GroupedWithOffset" in event["name"]
        ),
    )

    vocab_m_equivalent = 4 + 3
    vocab_flops = 2 * vocab_m_equivalent * HIDDEN * VOCAB_LOCAL
    vocab_bytes = 4 * HIDDEN * VOCAB_LOCAL * 2
    vocab_time = matching_time_us(traces, contains("nvjet_sm90_tst_192"))

    rows = [
        roofline_row(
            "Target fused QKV-A W4 Marlin",
            "HBM",
            LAYERS,
            qkv_time,
            qkv_flops,
            qkv_bytes,
        ),
        roofline_row(
            "Target W4 Machete projections/MLPs",
            "HBM",
            312,
            machete_time,
            machete_flops,
            machete_bytes,
        ),
        roofline_row(
            "Target MoE router BF16",
            "HBM",
            ROUTED_LAYERS,
            router_time,
            router_flops,
            router_bytes,
        ),
        roofline_row(
            "MLA W_UK/W_UV BF16 contractions",
            "HBM",
            2 * (LAYERS + 3),
            contractions_time,
            contractions_flops,
            contractions_bytes,
        ),
        roofline_row(
            "Target DSA Wq BF16",
            "HBM",
            21,
            wq_time,
            wq_flops,
            wq_bytes,
        ),
        roofline_row(
            "DSA WK+score projection BF16",
            "HBM",
            24,
            wk_time,
            wk_flops,
            wk_bytes,
        ),
        roofline_row(
            "DSA full-context FP8 scan",
            "HBM",
            24,
            index_time,
            index_flops,
            index_bytes,
            compute_ceiling=FP8_FLOPS_S,
            note="K-cache is counted once per batch-4 target call.",
        ),
        byte_row(
            "DSA cooperative top-k",
            "HBM",
            24,
            topk_time,
            topk_bytes,
            HBM_BYTES_S,
            "Minimum one FP32 logits read; selection work is not FLOP-modeled.",
        ),
        roofline_row(
            "Sparse MLA FP8 split+combine",
            "HBM",
            LAYERS + 3,
            mla_time,
            mla_flops,
            mla_bytes,
            note="Actual work includes 16-to-64 head padding; useful FLOPs are 25%.",
        ),
        roofline_row(
            "MTP eh_proj BF16",
            "HBM",
            3,
            eh_time,
            eh_flops,
            eh_bytes,
        ),
        roofline_row(
            "MTP FP8 dense/index/shared linears",
            "HBM",
            18,
            mtp_dense_time,
            mtp_dense_flops,
            mtp_dense_bytes,
            compute_ceiling=FP8_FLOPS_S,
        ),
        roofline_row(
            "MTP FP8 routed experts",
            "HBM",
            6,
            mtp_moe_time,
            mtp_moe_flops,
            mtp_moe_bytes,
            compute_ceiling=FP8_FLOPS_S,
            note="Uses the global average of two expert assignments per rank/pass.",
        ),
        roofline_row(
            "BF16 vocabulary projection",
            "HBM",
            4,
            vocab_time,
            vocab_flops,
            vocab_bytes,
        ),
    ]
    return rows


def routed_bounds(traces: list[dict]) -> dict:
    rank_stats = []
    for trace in traces:
        events = [
            event
            for event in kernel_events(trace)
            if "marlin_moe_wna16::Marlin" in event["name"]
        ]
        groups = [events[index : index + 4] for index in range(0, len(events), 4)]
        hot = [sum(event["dur"] for event in group[:2]) for group in groups]
        cold = [sum(event["dur"] for event in group[2:]) for group in groups]
        serial_us = sum(hot) + sum(cold)
        overlapped_us = sum(
            max(hot_us, cold_us) for hot_us, cold_us in zip(hot, cold, strict=True)
        )
        rank_stats.append(
            {
                "rank": trace["distributedInfo"]["rank"],
                "hot_us_per_step": sum(hot) / STEPS,
                "cold_us_per_step": sum(cold) / STEPS,
                "hot_active_layers_per_step": sum(value > 8 for value in hot) / STEPS,
                "cold_active_layers_per_step": sum(value > 8 for value in cold) / STEPS,
                "serial_us_per_step": serial_us / STEPS,
                "ideal_overlap_us_per_step": overlapped_us / STEPS,
                "overlap_upper_bound_savings_us": (serial_us - overlapped_us) / STEPS,
                "overlap_upper_bound_pct": 100
                * (serial_us - overlapped_us)
                / serial_us,
            }
        )

    hot_time = statistics.mean(item["hot_us_per_step"] for item in rank_stats)
    cold_time = statistics.mean(item["cold_us_per_step"] for item in rank_stats)
    cold_active = statistics.mean(
        item["cold_active_layers_per_step"] for item in rank_stats
    )
    total_assignments = ROUTED_LAYERS * 4 * 8 / 4
    cold_min = cold_active
    cold_max = min(
        total_assignments,
        cold_time * 1e-6 * C2C_BYTES_S / EXPERT_BYTES_W4,
    )
    hot_min = total_assignments - cold_max
    hot_max = total_assignments - cold_min

    mean_serial_us = statistics.mean(item["serial_us_per_step"] for item in rank_stats)
    mean_overlap_us = statistics.mean(
        item["ideal_overlap_us_per_step"] for item in rank_stats
    )

    def tier_range(
        assignments_min: float,
        assignments_max: float,
        time_us: float,
        ceiling: float,
    ) -> dict:
        bytes_min = assignments_min * EXPERT_BYTES_W4
        bytes_max = assignments_max * EXPERT_BYTES_W4
        flops_min = assignments_min * EXPERT_FLOPS
        flops_max = assignments_max * EXPERT_FLOPS
        seconds = time_us * 1e-6
        return {
            "assignments_per_rank_step": [assignments_min, assignments_max],
            "logical_bytes_per_step": [bytes_min, bytes_max],
            "effective_bandwidth_gbs": [
                bytes_min / seconds / 1e9,
                bytes_max / seconds / 1e9,
            ],
            "achieved_tflops": [
                flops_min / seconds / 1e12,
                flops_max / seconds / 1e12,
            ],
            "roof_efficiency_pct": [
                100 * bytes_min / seconds / ceiling,
                100 * bytes_max / seconds / ceiling,
            ],
        }

    return {
        "rank_stats": rank_stats,
        "method": (
            "Lower cold traffic is one distinct expert per active cold tier. "
            "Upper cold traffic is the measured 421 GB/s C2C ceiling. Remaining "
            "global-average assignments are attributed to HBM."
        ),
        "ideal_hot_cold_overlap": {
            "serial_us_per_step": mean_serial_us,
            "ideal_overlap_us_per_step": mean_overlap_us,
            "upper_bound_savings_us": mean_serial_us - mean_overlap_us,
            "upper_bound_savings_pct": 100
            * (mean_serial_us - mean_overlap_us)
            / mean_serial_us,
            "note": (
                "Kernel-only bound from overlapping each layer's hot and cold "
                "tier calls. End-to-end savings will be smaller."
            ),
        },
        "total_expert_assignments_per_rank_step": total_assignments,
        "hot_hbm": {
            "time_us_per_step": hot_time,
            **tier_range(hot_min, hot_max, hot_time, HBM_BYTES_S),
        },
        "cold_c2c": {
            "time_us_per_step": cold_time,
            **tier_range(cold_min, cold_max, cold_time, C2C_BYTES_S),
        },
    }


def communication(traces: list[dict]) -> dict:
    allreduce_time = matching_time_us(traces, contains("cross_device_reduce_1stage"))
    allreduce_calls = matching_calls(traces, contains("cross_device_reduce_1stage"))
    target_payload = 4 * HIDDEN * 2
    draft_payload = HIDDEN * 2
    remote_bytes = 3 * (157 * target_payload + 9 * draft_payload)
    ideal_us = 157 * 4.288506 + 9 * 4.058283

    allgather_time = matching_time_us(traces, contains("ncclDevKernel_AllGather"))
    target_chunk = 4 * VOCAB_LOCAL * 2
    draft_chunk = VOCAB_LOCAL * 2
    allgather_network_bytes = 3 * target_chunk + 3 * 3 * draft_chunk

    copies = defaultdict(lambda: {"calls": 0.0, "time_us": 0.0, "bytes": 0.0})
    for trace in traces:
        rank_copy = defaultdict(lambda: [0, 0.0, 0])
        for event in kernel_events(trace):
            if event["cat"] != "gpu_memcpy":
                continue
            item = rank_copy[event["name"]]
            item[0] += 1
            item[1] += event["dur"]
            item[2] += event.get("args", {}).get("bytes", 0)
        for name, (calls, time_us, byte_count) in rank_copy.items():
            copies[name]["calls"] += calls / STEPS / len(traces)
            copies[name]["time_us"] += time_us / STEPS / len(traces)
            copies[name]["bytes"] += byte_count / STEPS / len(traces)

    return {
        "custom_allreduce": {
            "calls_per_step": allreduce_calls,
            "time_us_per_step": allreduce_time,
            "remote_bytes_per_rank_step": remote_bytes,
            "effective_remote_gbs": remote_bytes / (allreduce_time * 1e-6) / 1e9,
            "link_bandwidth_efficiency_pct": 100
            * remote_bytes
            / (allreduce_time * 1e-6)
            / NVLINK_BYTES_S,
            "measured_latency_floor_us": ideal_us,
            "latency_efficiency_pct": 100 * ideal_us / allreduce_time,
        },
        "vocab_nccl_allgather": {
            "calls_per_step": 4,
            "time_us_per_step": allgather_time,
            "network_bytes_per_rank_step": allgather_network_bytes,
            "effective_gbs": allgather_network_bytes / (allgather_time * 1e-6) / 1e9,
            "link_bandwidth_efficiency_pct": 100
            * allgather_network_bytes
            / (allgather_time * 1e-6)
            / NVLINK_BYTES_S,
        },
        "explicit_copies": dict(copies),
    }


def write_inventory(path: Path, rows: list[dict]) -> None:
    fields = [
        "role",
        "category",
        "kernel",
        "calls_per_step",
        "time_us_per_step",
        "rank_min_us",
        "rank_max_us",
        "mean_call_us",
        "logical_bytes_per_step",
        "profiler_launch_occupancy_pct",
        "grids",
        "blocks",
        "streams",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_roofline(path: Path, rows: list[dict], routed: dict) -> None:
    points = [row for row in rows if row["achieved_tflops"] is not None]
    x_values = [10 ** (value / 80) for value in range(-24, 225)]
    bf16_roof = [min(630, 3.5 * value) for value in x_values]
    fp8_roof = [min(1260, 3.5 * value) for value in x_values]
    c2c_roof = [min(630, 0.421 * value) for value in x_values]

    figure, (roof_axis, efficiency_axis) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    roof_axis.loglog(x_values, bf16_roof, label="BF16 / HBM roof", color="#2667a5")
    roof_axis.loglog(
        x_values, fp8_roof, label="FP8 / HBM roof", color="#8631a8", linestyle="--"
    )
    roof_axis.loglog(x_values, c2c_roof, label="BF16 / Grace C2C roof", color="#d17516")

    abbreviations = {
        "Target fused QKV-A W4 Marlin": "W4 QKV-A",
        "Target W4 Machete projections/MLPs": "W4 Machete",
        "Target MoE router BF16": "Router",
        "MLA W_UK/W_UV BF16 contractions": "MLA BMM",
        "Target DSA Wq BF16": "DSA Wq",
        "DSA WK+score projection BF16": "DSA WK",
        "DSA full-context FP8 scan": "DSA scan",
        "Sparse MLA FP8 split+combine": "Sparse MLA",
        "MTP eh_proj BF16": "MTP eh",
        "MTP FP8 dense/index/shared linears": "MTP FP8",
        "MTP FP8 routed experts": "MTP MoE",
        "BF16 vocabulary projection": "Vocab",
    }
    label_offsets = {
        "Target MoE router BF16": (15, -13),
        "MLA W_UK/W_UV BF16 contractions": (4, 7),
        "Target DSA Wq BF16": (4, -13),
        "DSA WK+score projection BF16": (4, -12),
        "MTP eh_proj BF16": (4, 7),
        "MTP FP8 dense/index/shared linears": (4, 7),
        "MTP FP8 routed experts": (4, -11),
    }
    for row in points:
        x = row["arithmetic_intensity_flop_per_byte"]
        y = row["achieved_tflops"]
        roof_axis.scatter(x, y, s=45)
        roof_axis.annotate(
            abbreviations[row["name"]],
            (x, y),
            xytext=label_offsets.get(row["name"], (4, 4)),
            textcoords="offset points",
            fontsize=8,
        )

    for tier_name, label, color, offset in (
        ("hot_hbm", "Target MoE hot", "#2f8f46", (4, 8)),
        ("cold_c2c", "Target MoE cold", "#d17516", (4, 4)),
    ):
        tier = routed[tier_name]
        low, high = tier["achieved_tflops"]
        middle = math.sqrt(low * high)
        roof_axis.scatter(EXPERT_FLOPS / EXPERT_BYTES_W4, middle, s=60, color=color)
        roof_axis.vlines(
            EXPERT_FLOPS / EXPERT_BYTES_W4, low, high, color=color, linewidth=2
        )
        roof_axis.annotate(
            label,
            (EXPERT_FLOPS / EXPERT_BYTES_W4, middle),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )

    roof_axis.set_xlim(0.4, 500)
    roof_axis.set_ylim(0.2, 1600)
    roof_axis.set_xlabel("Arithmetic intensity (FLOP/byte)")
    roof_axis.set_ylabel("Achieved modeled throughput (TFLOP/s per rank)")
    roof_axis.set_title("Analytical GH200 roofline, one MTP target step")
    roof_axis.grid(True, which="both", alpha=0.2)
    roof_axis.legend(fontsize=8, loc="lower right")

    labels = [abbreviations[row["name"]] for row in points]
    efficiencies = [row["roof_efficiency_pct"] for row in points]
    positions = list(range(len(labels)))
    efficiency_axis.barh(positions, efficiencies, color="#4f86b5")
    efficiency_axis.set_yticks(positions, labels)
    efficiency_axis.invert_yaxis()
    efficiency_axis.set_xlim(0, 100)
    efficiency_axis.set_xlabel("Efficiency against applicable roof (%)")
    efficiency_axis.set_title("Modeled kernel-family efficiency")
    efficiency_axis.grid(True, axis="x", alpha=0.2)
    for position, value in zip(positions, efficiencies, strict=True):
        efficiency_axis.text(
            min(value + 1, 96), position, f"{value:.1f}%", va="center", fontsize=8
        )

    figure.tight_layout()
    figure.savefig(path, format="svg")
    figure.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(figure)
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    traces = load_traces(args.trace_dir)
    rows, role_times = inventory(traces)
    roofline = modeled_rows(traces)
    routed = routed_bounds(traces)
    comms = communication(traces)
    total_activity = statistics.mean(
        sum(event["dur"] for event in kernel_events(trace)) / STEPS for trace in traces
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_inventory(args.output_dir / "roofline-kernels.csv", rows)
    summary = {
        "method": {
            "profiled_steps": STEPS,
            "query_size": 4,
            "context_tokens": CONTEXT,
            "trace_directory": str(args.trace_dir),
            "trace_files": [
                path.name for path in sorted(args.trace_dir.glob("*.trace.json.gz"))
            ],
            "trace_has_hardware_counters": False,
            "duration_source": "PyTorch profiler CUDA event durations",
            "traffic_source": "checkpoint shapes, model geometry, and kernel I/O",
        },
        "ceilings": {
            "hbm_bytes_per_second": HBM_BYTES_S,
            "grace_c2c_measured_bytes_per_second": C2C_BYTES_S,
            "peer_nvlink_bytes_per_second": NVLINK_BYTES_S,
            "bf16_flops_per_second": BF16_FLOPS_S,
            "fp8_flops_per_second": FP8_FLOPS_S,
        },
        "gpu_kernel_activity_us_per_step": total_activity,
        "activity_by_role_us_per_step": role_times,
        "roofline": roofline,
        "routed_moe_bounds": routed,
        "communication": comms,
    }
    (args.output_dir / "roofline-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    plot_roofline(args.output_dir / "roofline.svg", roofline, routed)


if __name__ == "__main__":
    main()
