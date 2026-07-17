#!/usr/bin/env python3

import argparse
import ctypes
import json
import os
import statistics
from collections import Counter
from dataclasses import asdict, dataclass

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


@dataclass
class Result:
    source: str
    experts: int
    median_group_us: float
    median_expert_us: float
    p90_group_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--warmup-groups", type=int, default=10)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--numa-node", type=int, default=0)
    return parser.parse_args()


def allocate_grace_copy(tensor: torch.Tensor, numa_node: int) -> GraceAllocation:
    allocation = GraceAllocation.allocate_pinned(
        tuple(tensor.shape), tensor.dtype, tensor.device.index or 0, numa_node
    )
    allocation.copy_from(tensor)
    return allocation


def sample_numa_nodes(tensor: torch.Tensor, samples: int = 256) -> dict[str, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    num_bytes = tensor.numel() * tensor.element_size()
    step = max(page_size, num_bytes // samples)
    step -= step % page_size
    offsets = list(range(0, num_bytes, step))[:samples]
    pages = (ctypes.c_void_p * len(offsets))(
        *(tensor.data_ptr() + offset for offset in offsets)
    )
    status = (ctypes.c_int * len(offsets))()
    libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    result = libnuma.move_pages(0, len(offsets), pages, None, status, 0)
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return dict(sorted(Counter(str(node) for node in status).items()))


def run_group(
    activation: torch.Tensor,
    weights: list[torch.Tensor],
    scales: list[torch.Tensor],
    workspace: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    output = torch.empty(0, device=activation.device)
    empty = torch.empty(0, dtype=torch.int32, device=activation.device)
    for weight, scale in zip(weights, scales, strict=True):
        output = ops.marlin_gemm(
            activation,
            None,
            weight,
            None,
            scale,
            None,
            None,
            None,
            empty,
            empty,
            workspace,
            scalar_types.uint4b8,
            args.m,
            args.n,
            args.k,
            True,
            False,
            False,
            False,
        )
    return output


def measure_interleaved(
    activation: torch.Tensor,
    hbm_weights: list[torch.Tensor],
    hbm_scales: list[torch.Tensor],
    grace_weights: list[torch.Tensor],
    grace_scales: list[torch.Tensor],
    workspace: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[Result, Result, torch.Tensor, torch.Tensor]:
    sources = {
        "hbm": (hbm_weights, hbm_scales),
        "pinned_grace_uva": (grace_weights, grace_scales),
    }
    outputs: dict[str, torch.Tensor] = {}
    for _ in range(args.warmup_groups):
        for source, (weights, scales) in sources.items():
            outputs[source] = run_group(activation, weights, scales, workspace, args)
    torch.cuda.synchronize()

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        source: [] for source in sources
    }
    source_names = list(sources)
    for group_index in range(args.groups):
        order = source_names if group_index % 2 else reversed(source_names)
        for source in order:
            weights, scales = sources[source]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs[source] = run_group(activation, weights, scales, workspace, args)
            end.record()
            events[source].append((start, end))
    torch.cuda.synchronize()

    results = []
    for source, source_events in events.items():
        times_us = [start.elapsed_time(end) * 1000 for start, end in source_events]
        median_group_us = statistics.median(times_us)
        results.append(
            Result(
                source=source,
                experts=args.experts,
                median_group_us=median_group_us,
                median_expert_us=median_group_us / args.experts,
                p90_group_us=sorted(times_us)[int(0.9 * (len(times_us) - 1))],
            )
        )
    return (
        results[0],
        results[1],
        outputs["hbm"],
        outputs["pinned_grace_uva"],
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    activation = torch.randn(args.m, args.k, dtype=torch.float16, device=device)
    dense = torch.randn(args.k, args.n, dtype=torch.float16, device=device)
    _, packed, scale, _, _, _ = marlin_quantize(
        dense, scalar_types.uint4b8, args.group_size, False
    )
    del dense

    hbm_weights = [packed.clone() for _ in range(args.experts)]
    hbm_scales = [scale.clone() for _ in range(args.experts)]
    del packed, scale
    grace_weights = [
        allocate_grace_copy(weight, args.numa_node) for weight in hbm_weights
    ]
    grace_scales = [allocate_grace_copy(scale, args.numa_node) for scale in hbm_scales]
    workspace = marlin_make_workspace_new(device)
    grace_numa_nodes_before = sample_numa_nodes(grace_weights[0].cpu_tensor)

    hbm_result, grace_result, hbm_output, grace_output = measure_interleaved(
        activation,
        hbm_weights,
        hbm_scales,
        [allocation.cuda_alias for allocation in grace_weights],
        [allocation.cuda_alias for allocation in grace_scales],
        workspace,
        args,
    )
    torch.testing.assert_close(grace_output, hbm_output, rtol=0, atol=0)

    report = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "shape_mkn": [args.m, args.k, args.n],
        "group_size": args.group_size,
        "packed_bytes_per_expert": grace_weights[0].num_bytes,
        "scale_bytes_per_expert": grace_scales[0].num_bytes,
        "results": [asdict(hbm_result), asdict(grace_result)],
        "grace_over_hbm": (grace_result.median_group_us / hbm_result.median_group_us),
        "grace_numa_nodes_before": grace_numa_nodes_before,
        "grace_numa_nodes_after": sample_numa_nodes(grace_weights[0].cpu_tensor),
        "correctness": "exact",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
