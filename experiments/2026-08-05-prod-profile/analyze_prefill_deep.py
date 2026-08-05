"""Deep analysis of one 8,192-token prefill chunk.

Restricts to the prefill steps, then prices each family against what the
hardware could do: MoE GEMM FLOPs against the bf16 tensor-core peak, collective
bytes against the NVLink ring bound, and weight traffic against the C2C and HBM
roofs. The point is to separate "big because the work is big" from "big because
the kernel or protocol is wrong".
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

SMEM_DIR = Path(__file__).resolve().parents[1] / "2026-07-29-marlin-smem-monopoly"
sys.path.insert(0, str(SMEM_DIR))

from analyze_step_budget import (  # noqa: E402
    GPU_CATEGORIES,
    family,
    load_trace,
    step_windows,
)

# GLM-5.2 shapes.
HIDDEN = 6144
MOE_INTERMEDIATE = 2048
TOPK = 8
ROUTED_LAYERS = 75
EXPERTS_PER_RANK = 64
BYTES_PER_EXPERT = 20_054_024  # W4G64 layer-expert copy, from the physical plan

# Machine.
EP = 4
BF16_PEAK_TFLOPS = 989.0  # GH200 SXM dense bf16
HBM_GBS = 3350.0
C2C_GBS = 421.0  # measured achievable, 2026-07-25-grace-bandwidth
PREFILL_STEP_MS = 200.0


def bucket(name: str) -> str:
    if "Marlin" in name:
        return "routed MoE (W4 Marlin)"
    if "nccl" in name:
        for collective in ("AllReduce", "AllGather", "ReduceScatter", "SendRecv"):
            if collective in name:
                return f"NCCL {collective}"
        return "NCCL other"
    if "cross_device_reduce" in name:
        return "custom all-reduce"
    if "mla" in name or "flash" in name:
        return "attention (MLA)"
    if "sparse_attn_indexer" in name or "topk" in name or "index" in name.lower():
        return "attention (DSA index)"
    if "nvjet" in name or "triton_tem" in name or "cutlass" in name:
        return "dense GEMM"
    return family(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    trace = sorted(args.capture.glob("*rank0*.trace.json.gz"))[0]
    events = load_trace(trace)
    windows = [w for w in step_windows(events) if w[1] - w[0] >= PREFILL_STEP_MS]
    if not windows:
        raise SystemExit("no prefill-sized steps in this capture")
    chunks = len(windows)

    def in_prefill(timestamp: float) -> bool:
        return any(start <= timestamp < end for start, end in windows)

    gpu = [
        event
        for event in events
        if event.get("cat") in GPU_CATEGORIES and in_prefill(event["t"])
    ]
    by_bucket: dict[str, float] = collections.defaultdict(float)
    counts: collections.Counter = collections.Counter()
    kernels: dict[str, float] = collections.defaultdict(float)
    occupancy: dict[str, list] = collections.defaultdict(list)
    for event in gpu:
        name = event["name"]
        key = bucket(name)
        by_bucket[key] += event["dur"] / 1000
        counts[key] += 1
        kernels[name[:70]] += event["dur"] / 1000
        if "Marlin" in name:
            occupancy["grid"].append(event.get("args", {}).get("grid"))
            occupancy["occ"].append(
                event.get("args", {}).get("est. achieved occupancy %")
            )
            occupancy["smem"].append(event.get("args", {}).get("shared memory"))

    # Collective payloads, from the CPU op that record_shapes annotated.
    payloads: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for event in events:
        if event.get("cat") != "cpu_op" or not in_prefill(event["t"]):
            continue
        dims = event.get("args", {}).get("Input Dims")
        if not dims or "all_reduce" not in event["name"]:
            continue
        shape = dims[0]
        if isinstance(shape, list) and len(shape) == 2:
            payloads[event["name"]][tuple(shape)] += 1

    span_ms = sum(end - start for start, end in windows)
    print(f"trace {trace.name}")
    print(f"{chunks} prefill chunk step(s), {span_ms / chunks:.1f} ms each\n")

    print("per chunk, by bucket:")
    total = sum(by_bucket.values())
    for key, value in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
        print(
            f"  {key:28} {value / chunks:9.1f} ms  {value / total * 100:5.1f}%"
            f"  x{counts[key] / chunks:7.1f}"
        )
    print(f"  {'GPU busy total':28} {total / chunks:9.1f} ms")
    print(f"  {'step wall':28} {span_ms / chunks:9.1f} ms")

    print("\ntop kernels per chunk:")
    for name, value in sorted(kernels.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {value / chunks:9.1f} ms  {name}")

    print("\ncollective payloads (CPU-op recorded shapes):")
    for name, shapes in payloads.items():
        for shape, count in shapes.most_common(3):
            nbytes = shape[0] * shape[1] * 2
            print(f"  {name}  {shape}  x{count}  {nbytes / 2**20:.1f} MiB each")

    # --- Roofline checks ---
    print("\n--- roofline ---")

    # MoE GEMM: every token visits TOPK experts; this rank owns EP_size^-1 of them.
    assignments = 8192 * TOPK / EP
    flops_per_assignment = 2 * (
        HIDDEN * 2 * MOE_INTERMEDIATE + MOE_INTERMEDIATE * HIDDEN
    )
    moe_tflop = assignments * flops_per_assignment * ROUTED_LAYERS / 1e12
    moe_ms = by_bucket["routed MoE (W4 Marlin)"] / chunks
    achieved = moe_tflop / (moe_ms / 1000)
    print(f"routed MoE: {moe_tflop:.1f} TFLOP/rank/chunk in {moe_ms:.1f} ms")
    print(
        f"  = {achieved:.1f} TFLOP/s, {achieved / BF16_PEAK_TFLOPS * 100:.1f}% of "
        f"the {BF16_PEAK_TFLOPS:.0f} TFLOP/s bf16 peak"
    )

    # Weight traffic, if every owned expert is touched once per layer.
    weight_gb = EXPERTS_PER_RANK * BYTES_PER_EXPERT * ROUTED_LAYERS / 1e9
    hot_share = 2870 / 4800
    hot_ms = weight_gb * hot_share / HBM_GBS * 1000
    cold_ms = weight_gb * (1 - hot_share) / C2C_GBS * 1000
    print(
        f"weight streaming floor: {weight_gb:.1f} GB/rank/chunk "
        f"= {hot_ms:.0f} ms HBM + {cold_ms:.0f} ms C2C = {hot_ms + cold_ms:.0f} ms"
    )
    print(f"  so MoE is compute-bound in prefill by {moe_ms / (hot_ms + cold_ms):.1f}x")

    # Collectives against the ring bound.
    for key in ("NCCL AllReduce", "NCCL AllGather", "NCCL ReduceScatter"):
        if key not in by_bucket:
            continue
        ms = by_bucket[key] / chunks
        n = counts[key] / chunks
        size_mib = 8192 * HIDDEN * 2 / 2**20
        factor = {"NCCL AllReduce": 2 * (EP - 1) / EP}.get(key, (EP - 1) / EP)
        bus_gb = n * size_mib * 2**20 * factor / 1e9
        print(
            f"{key}: {ms:.1f} ms over {n:.0f} calls; if each moves "
            f"{size_mib:.0f} MiB, bus bandwidth = {bus_gb / (ms / 1000):.0f} GB/s"
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "chunks": chunks,
                    "chunk_wall_ms": span_ms / chunks,
                    "buckets_ms": {k: v / chunks for k, v in by_bucket.items()},
                    "counts": {k: v / chunks for k, v in counts.items()},
                    "kernels_ms": {k: v / chunks for k, v in kernels.items()},
                    "moe_tflop": moe_tflop,
                    "moe_achieved_tflops": achieved,
                    "weight_stream_floor_ms": hot_ms + cold_ms,
                },
                indent=2,
                default=float,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
