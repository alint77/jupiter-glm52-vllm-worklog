#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.offloader.grace import GraceAllocation
from vllm.scalar_type import scalar_types


@dataclass(frozen=True)
class Result:
    name: str
    median_us: float
    p90_us: float
    amortized_wall_us: float


@dataclass
class Workspaces:
    marlin: torch.Tensor
    intermediate13: torch.Tensor
    intermediate2: torch.Tensor
    output: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hot-experts", type=int, default=44)
    parser.add_argument("--cold-experts", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--m", type=int, default=1)
    return parser.parse_args()


def make_expert_map(start: int, count: int, device: torch.device) -> torch.Tensor:
    expert_map = torch.full((256,), -1, dtype=torch.int32, device=device)
    expert_map[start : start + count] = torch.arange(
        count, dtype=torch.int32, device=device
    )
    return expert_map


def make_workspaces(device: torch.device, num_tokens: int) -> Workspaces:
    return Workspaces(
        marlin=marlin_make_workspace_new(device, 4),
        intermediate13=torch.empty(
            num_tokens * 8 * 6144, dtype=torch.bfloat16, device=device
        ),
        intermediate2=torch.empty(
            (num_tokens * 8, 2048), dtype=torch.bfloat16, device=device
        ),
        output=torch.empty(
            (num_tokens, 6144), dtype=torch.bfloat16, device=device
        ),
    )


def run_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor,
    workspaces: Workspaces,
) -> torch.Tensor:
    return fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w13,
        w2=w2,
        bias1=None,
        bias2=None,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=scalar_types.uint4b8.id,
        global_num_experts=256,
        activation=MoEActivation.SILU,
        expert_map=expert_map,
        workspace=workspaces.marlin,
        intermediate_cache13=workspaces.intermediate13,
        intermediate_cache2=workspaces.intermediate2,
        output=workspaces.output,
        is_k_full=True,
    )


def measure(
    name: str,
    function: Callable[[], torch.Tensor],
    warmups: int,
    iterations: int,
) -> tuple[Result, torch.Tensor]:
    for _ in range(warmups):
        output = function()
    torch.cuda.synchronize()

    events = []
    wall_start = time.perf_counter()
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    amortized_wall_us = (time.perf_counter() - wall_start) * 1_000_000 / iterations
    times_us = [start.elapsed_time(end) * 1000 for start, end in events]
    return (
        Result(
            name=name,
            median_us=statistics.median(times_us),
            p90_us=sorted(times_us)[int(0.9 * (len(times_us) - 1))],
            amortized_wall_us=amortized_wall_us,
        ),
        output,
    )


def main() -> None:
    args = parse_args()
    if args.hot_experts + args.cold_experts != 64:
        raise ValueError("Hot and cold experts must total one EP4 rank")
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    hidden_states = torch.randn(
        (args.m, 6144), dtype=torch.bfloat16, device=device
    )
    topk_weights = torch.full(
        (args.m, 8), 0.125, dtype=torch.float32, device=device
    )

    w13 = torch.empty((64, 384, 8192), dtype=torch.int32, device=device)
    w2 = torch.empty((64, 128, 12288), dtype=torch.int32, device=device)
    w13.random_()
    w2.random_()
    w13_scale = torch.rand(
        (64, 48, 4096), dtype=torch.bfloat16, device=device
    )
    w2_scale = torch.rand(
        (64, 16, 6144), dtype=torch.bfloat16, device=device
    )

    hot = slice(0, args.hot_experts)
    cold = slice(args.hot_experts, 64)
    native_map = make_expert_map(0, 64, device)
    hot_map = make_expert_map(0, args.hot_experts, device)
    cold_map = make_expert_map(args.hot_experts, args.cold_experts, device)

    grace_allocations = [
        GraceAllocation.allocate_pinned(
            tuple(tensor.shape), tensor.dtype, 0, args.numa_node
        )
        for tensor in (w13[cold], w2[cold], w13_scale[cold], w2_scale[cold])
    ]
    for allocation, source in zip(
        grace_allocations,
        (w13[cold], w2[cold], w13_scale[cold], w2_scale[cold]),
        strict=True,
    ):
        allocation.copy_from(source)
    grace_w13, grace_w2, grace_w13_scale, grace_w2_scale = (
        allocation.cuda_alias for allocation in grace_allocations
    )
    native_grace_allocations = [
        GraceAllocation.allocate_pinned(
            tuple(tensor.shape), tensor.dtype, 0, args.numa_node
        )
        for tensor in (w13, w2, w13_scale, w2_scale)
    ]
    for allocation, source in zip(
        native_grace_allocations,
        (w13, w2, w13_scale, w2_scale),
        strict=True,
    ):
        allocation.copy_from(source)
    native_grace_tensors = tuple(
        allocation.cuda_alias for allocation in native_grace_allocations
    )

    native_workspace = make_workspaces(device, args.m)
    hot_workspace = make_workspaces(device, args.m)
    cold_workspace = make_workspaces(device, args.m)
    mixed_ids = torch.tensor(
        [[0, args.hot_experts, 64, 65, 66, 67, 68, 69]],
        dtype=torch.int32,
        device=device,
    ).repeat(args.m, 1)
    hot_ids = torch.tensor(
        [[0, 1, 64, 65, 66, 67, 68, 69]], dtype=torch.int32, device=device
    ).repeat(args.m, 1)
    cold_ids = torch.tensor(
        [
            [
                args.hot_experts,
                args.hot_experts + 1,
                64,
                65,
                66,
                67,
                68,
                69,
            ]
        ],
        dtype=torch.int32,
        device=device,
    ).repeat(args.m, 1)
    cycling_ids = [
        torch.tensor(
            [[0, args.hot_experts + index, 64, 65, 66, 67, 68, 69]],
            dtype=torch.int32,
            device=device,
        )
        for index in range(args.cold_experts)
    ]
    cycling_ids = [topk_ids.repeat(args.m, 1) for topk_ids in cycling_ids]
    native_cycling_ids = [
        torch.tensor(
            [[index, (index + 1) % 64, 64, 65, 66, 67, 68, 69]],
            dtype=torch.int32,
            device=device,
        )
        for index in range(64)
    ]
    native_cycling_ids = [
        topk_ids.repeat(args.m, 1) for topk_ids in native_cycling_ids
    ]

    def native() -> torch.Tensor:
        return run_moe(
            hidden_states,
            w13,
            w2,
            w13_scale,
            w2_scale,
            mixed_ids,
            topk_weights,
            native_map,
            native_workspace,
        )

    def native_grace(topk_ids: torch.Tensor) -> torch.Tensor:
        return run_moe(
            hidden_states,
            *native_grace_tensors,
            topk_ids,
            topk_weights,
            native_map,
            native_workspace,
        )

    def split(topk_ids: torch.Tensor, grace: bool) -> torch.Tensor:
        hot_output = run_moe(
            hidden_states,
            w13[hot],
            w2[hot],
            w13_scale[hot],
            w2_scale[hot],
            topk_ids,
            topk_weights,
            hot_map,
            hot_workspace,
        )
        cold_tensors = (
            (grace_w13, grace_w2, grace_w13_scale, grace_w2_scale)
            if grace
            else (w13[cold], w2[cold], w13_scale[cold], w2_scale[cold])
        )
        cold_output = run_moe(
            hidden_states,
            *cold_tensors,
            topk_ids,
            topk_weights,
            cold_map,
            cold_workspace,
        )
        return hot_output + cold_output

    def make_cycling_case(grace: bool) -> Callable[[], torch.Tensor]:
        index = 0

        def cycling_case() -> torch.Tensor:
            nonlocal index
            topk_ids = cycling_ids[index]
            index = (index + 1) % len(cycling_ids)
            return split(topk_ids, grace)

        return cycling_case

    def make_native_cycling_case(grace: bool) -> Callable[[], torch.Tensor]:
        index = 0

        def cycling_case() -> torch.Tensor:
            nonlocal index
            topk_ids = native_cycling_ids[index]
            index = (index + 1) % len(native_cycling_ids)
            if grace:
                return native_grace(topk_ids)
            return run_moe(
                hidden_states,
                w13,
                w2,
                w13_scale,
                w2_scale,
                topk_ids,
                topk_weights,
                native_map,
                native_workspace,
            )

        return cycling_case

    cases = {
        "native_hbm_mixed": native,
        "native_grace_mixed": lambda: native_grace(mixed_ids),
        "native_hbm_cycling": make_native_cycling_case(False),
        "native_grace_cycling": make_native_cycling_case(True),
        "split_hbm_mixed": lambda: split(mixed_ids, False),
        "split_grace_mixed": lambda: split(mixed_ids, True),
        "split_hbm_cycling_cold": make_cycling_case(False),
        "split_grace_cycling_cold": make_cycling_case(True),
        "split_grace_hot_only": lambda: split(hot_ids, True),
        "split_grace_cold_only": lambda: split(cold_ids, True),
    }
    results = []
    outputs = {}
    for name, function in cases.items():
        result, output = measure(name, function, args.warmups, args.iterations)
        results.append(result)
        outputs[name] = output.clone()

    torch.testing.assert_close(
        outputs["split_hbm_mixed"],
        outputs["native_hbm_mixed"],
        rtol=0.03,
        atol=4096,
    )
    torch.testing.assert_close(
        outputs["split_grace_mixed"],
        outputs["native_hbm_mixed"],
        rtol=0.03,
        atol=4096,
    )
    torch.testing.assert_close(
        outputs["native_grace_mixed"],
        outputs["native_hbm_mixed"],
        rtol=0.03,
        atol=4096,
    )
    all_grace_allocations = grace_allocations + native_grace_allocations
    report = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "hot_experts": args.hot_experts,
        "cold_experts": args.cold_experts,
        "num_tokens": args.m,
        "grace_bytes": sum(
            allocation.num_bytes for allocation in all_grace_allocations
        ),
        "grace_locality": [
            asdict(allocation.audit_numa()) for allocation in all_grace_allocations
        ],
        "results": [asdict(result) for result in results],
        "correctness": "split HBM and Grace mixed outputs match native HBM",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
