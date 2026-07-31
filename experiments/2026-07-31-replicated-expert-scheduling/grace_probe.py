#!/usr/bin/env python3
"""Allocate, touch, audit, and read candidate pinned-Grace footprints."""

import argparse
import json
import time
from pathlib import Path

import torch

from vllm.model_executor.offloader.grace import GraceAllocation


def node_free_bytes(node: int) -> int:
    path = Path(f"/sys/devices/system/node/node{node}/meminfo")
    for line in path.read_text().splitlines():
        if "MemFree:" in line:
            return int(line.split()[-2]) * 1024
    raise RuntimeError(f"MemFree is missing from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numa-node", type=int, required=True)
    parser.add_argument("--chunk-mb", type=int, default=1024)
    parser.add_argument("--targets-gb", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    chunk_bytes = args.chunk_mb * 1024**2
    allocations = []
    rows = []
    allocated = 0
    for target_gb in args.targets_gb:
        target = round(target_gb * 1e9)
        started = time.perf_counter()
        error = None
        try:
            while allocated < target:
                size = min(chunk_bytes, target - allocated)
                allocation = GraceAllocation.allocate_pinned(
                    (size,),
                    torch.uint8,
                    args.device,
                    args.numa_node,
                )
                allocation.cpu_tensor.zero_()
                allocations.append(allocation)
                allocated += size
            audits = [
                allocations[index].audit_numa(samples=16, strict=True)
                for index in {0, len(allocations) // 2, len(allocations) - 1}
            ]
            destination = torch.empty(
                min(chunk_bytes, allocated),
                dtype=torch.uint8,
                device=f"cuda:{args.device}",
            )
            torch.cuda.synchronize()
            read_started = time.perf_counter()
            for allocation in allocations:
                size = allocation.num_bytes
                destination[:size].copy_(allocation.cuda_alias, non_blocking=True)
            torch.cuda.synchronize()
            read_seconds = time.perf_counter() - read_started
        except (RuntimeError, OSError) as exc:
            audits = []
            read_seconds = None
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "target_gb": target_gb,
                "allocated_gb": allocated / 1e9,
                "allocation_seconds": time.perf_counter() - started,
                "node_free_gb": node_free_bytes(args.numa_node) / 1e9,
                "local_fractions": [audit.local_fraction for audit in audits],
                "c2c_read_gb_s": (
                    allocated / read_seconds / 1e9 if read_seconds else None
                ),
                "error": error,
            }
        )
        if error:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "device": args.device,
                "numa_node": args.numa_node,
                "chunk_mb": args.chunk_mb,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output.read_text())


if __name__ == "__main__":
    main()
