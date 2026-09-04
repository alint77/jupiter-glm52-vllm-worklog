#!/usr/bin/env python3
"""Phase 1 of the cold-expert prefetch: is the Grace->HBM copy fast and quiet?

The design in `experiments/2026-09-05-cold-prefetch/PLAN.md` assumes a staging
copy that (a) sustains >=400 GB/s and (b) does not meaningfully disturb the
compute it runs beside. Both are assumptions until measured here, and either
one failing kills the design cheaply.

Three transfer mechanisms are compared, because they are not the same thing and
the project's 373 GB/s figure was measured on the third:

* `h2d_pinned`  -- `cudaMemcpyAsync` from the pinned CPU tensor. This is what
  the prefetch would issue, and it uses the copy engine.
* `d2d_uva`     -- a device-to-device copy from the Grace allocation's CUDA
  alias, which is how the tier storage is addressed today.
* `sm_read`     -- an SM-issued elementwise read of the alias, the mechanism
  `pageable_grace_bandwidth.py` measured.

Then Marlin runs alone and again with the copy concurrent on a second stream,
which gives both the copy rate under load and the compute slowdown.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.model_executor.offloader.grace import GraceAllocation
from vllm.scalar_type import scalar_types

MIB = 1 << 20


@dataclass
class Marlin:
    """A stand-in for one layer's routed MoE work, all weights in HBM."""

    activation: torch.Tensor
    weights: list[torch.Tensor]
    scales: list[torch.Tensor]
    workspace: torch.Tensor
    m: int
    n: int
    k: int
    empty: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.empty = torch.empty(0, dtype=torch.int32, device=self.activation.device)

    def run(self) -> None:
        for weight, scale in zip(self.weights, self.scales, strict=True):
            ops.marlin_gemm(
                self.activation, None, weight, None, scale, None, None, None,
                self.empty, self.empty, self.workspace,
                scalar_types.uint4b8, self.m, self.n, self.k,
                True, False, False, False,
            )


def build_marlin(experts: int, m: int, k: int, n: int, device: torch.device) -> Marlin:
    activation = torch.randn((m, k), dtype=torch.half, device=device)
    weights, scales = [], []
    for _ in range(experts):
        base = torch.randn((k, n), dtype=torch.half, device=device)
        _, packed, scale, _, _, _ = marlin_quantize(
            base, scalar_types.uint4b8, 128, False
        )
        weights.append(packed)
        scales.append(scale)
    return Marlin(activation, weights, scales,
                  marlin_make_workspace_new(device, 4), m, n, k)


def time_stream(stream: torch.cuda.Stream, body, iterations: int) -> float:
    """Median-free mean elapsed ms of `body` on `stream`, events on that stream."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iterations):
            body()
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numa-node", type=int, required=True)
    ap.add_argument("--layer-mib", type=int, nargs="*", default=[670, 830],
                    help="median and largest cold layer from the 2026-09-04 profile")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--m", type=int, default=256, help="routed rows per expert")
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out")
    args = ap.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    copy_stream = torch.cuda.Stream(device=device)
    report: dict = {"numa_node": args.numa_node, "sizes": {}}

    print(f"device {torch.cuda.get_device_name(device)}, NUMA node {args.numa_node}")
    marlin = build_marlin(args.experts, args.m, args.k, args.n, device)
    for _ in range(args.warmup):
        marlin.run()
    torch.cuda.synchronize()
    solo_ms = time_stream(torch.cuda.current_stream(device), marlin.run, args.iterations)
    print(f"Marlin alone ({args.experts} experts, m={args.m} k={args.k} n={args.n}): "
          f"{solo_ms:.3f} ms")
    report["marlin_solo_ms"] = solo_ms
    report["marlin"] = {"experts": args.experts, "m": args.m, "k": args.k, "n": args.n}

    for mib in args.layer_mib:
        num_bytes = mib * MIB
        grace = GraceAllocation.allocate_pinned(
            (num_bytes,), torch.uint8, 0, args.numa_node
        )
        grace.cpu_tensor.fill_(7)
        placement = grace.audit_numa(strict=False)
        staging = torch.empty(num_bytes, dtype=torch.uint8, device=device)
        scratch = torch.empty(num_bytes, dtype=torch.uint8, device=device)

        def h2d() -> None:
            staging.copy_(grace.cpu_tensor, non_blocking=True)

        def d2d() -> None:
            staging.copy_(grace.cuda_alias, non_blocking=True)

        def sm_read() -> None:
            torch.add(grace.cuda_alias, 1, out=scratch)

        entry: dict = {"bytes": num_bytes,
                       "numa_local_fraction": placement.local_fraction,
                       "numa_pages": list(placement.page_counts)}
        for name, body in (("h2d_pinned", h2d), ("d2d_uva", d2d), ("sm_read", sm_read)):
            for _ in range(args.warmup):
                body()
            torch.cuda.synchronize()
            ms = time_stream(copy_stream, body, args.iterations)
            entry[name] = {"ms": ms, "gbps": num_bytes / (ms / 1000) / 1e9}
            print(f"  {mib:4d} MiB {name:11s} {ms:7.3f} ms  "
                  f"{entry[name]['gbps']:6.0f} GB/s")

        # concurrent: Marlin on the compute stream, the copy on its own
        compute = torch.cuda.current_stream(device)
        torch.cuda.synchronize()
        start_c = torch.cuda.Event(enable_timing=True)
        end_c = torch.cuda.Event(enable_timing=True)
        start_k = torch.cuda.Event(enable_timing=True)
        end_k = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(copy_stream):
            start_c.record(copy_stream)
        start_k.record(compute)
        for _ in range(args.iterations):
            marlin.run()
        end_k.record(compute)
        with torch.cuda.stream(copy_stream):
            for _ in range(args.iterations):
                h2d()
            end_c.record(copy_stream)
        torch.cuda.synchronize()
        conc_kernel = start_k.elapsed_time(end_k) / args.iterations
        conc_copy = start_c.elapsed_time(end_c) / args.iterations
        entry["concurrent"] = {
            "marlin_ms": conc_kernel,
            "marlin_slowdown_pct": 100 * (conc_kernel - solo_ms) / solo_ms,
            "copy_ms": conc_copy,
            "copy_gbps": num_bytes / (conc_copy / 1000) / 1e9,
            "copy_gbps_solo_pct": 100 * (entry["h2d_pinned"]["ms"] / conc_copy),
        }
        print(f"  {mib:4d} MiB concurrent: Marlin {conc_kernel:7.3f} ms "
              f"({entry['concurrent']['marlin_slowdown_pct']:+5.1f}%), "
              f"copy {entry['concurrent']['copy_gbps']:6.0f} GB/s "
              f"({entry['concurrent']['copy_gbps_solo_pct']:.0f}% of solo)")
        print(f"  {mib:4d} MiB NUMA local fraction {placement.local_fraction:.3f}")
        report["sizes"][str(mib)] = entry
        del staging, scratch, grace
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
