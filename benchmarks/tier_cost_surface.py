#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure the tiered-MoE cost surface t(hot, cold, tokens) under graph replay.

The placement optimiser prices a layer as a linear cost in cold experts, and
`tier_balance.py` extends that to `max(t_hot * H, t_cold * C)` with the two
constants Phase 32 fitted (9.75 and 45.32 us). Those came from a regression
against per-layer active-expert counts in a single production trace, so they are
local slopes at one operating point, and three mechanisms say the surface is not
a plane:

  * the tight shared-memory policy that lets the tiers overlap makes each tier's
    kernel slower in isolation, so a tier's cost depends on whether the other is
    co-resident;
  * Marlin moves from weight-bandwidth-bound toward compute-bound as the token
    count rises, and the hot tier feels that while the C2C-bound cold tier does
    not, so the ratio between them is a function of m;
  * both tiers launch a fixed grid (SMs x 2 for hot, x 1 for cold), so per-expert
    cost is quantised by how the expert count divides into that grid. (4 hot,
    1 cold) and (8 hot, 2 cold) share a ratio but need not share a per-expert
    cost.

This measures the three quantities the model needs -- each tier alone, and the
two-stream union as production launches them -- over a grid of (H, C, m).

Timing is under CUDA graph replay. Timing a fork/join eagerly charges two stream
barriers per iteration, a ~110 us floor on Booster that would swamp the effect.
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    MarlinLaunchPolicy,
    fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.offloader.grace import GraceAllocation
from vllm.scalar_type import scalar_types

HIDDEN = 6144
INTERMEDIATE = 2048
TOPK = 8
NUM_EXPERTS = 256
# GLM-5.3 W4A16 group 32, the deployed checkpoint: 21,233,664 bytes per expert.
GROUP_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hot", type=int, nargs="+", default=[2, 4, 8, 12, 16, 20, 24])
    parser.add_argument(
        "--cold", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8, 12]
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--overlap-max-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@dataclass
class Tier:
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    expert_map: torch.Tensor
    workspace: torch.Tensor
    cache13: torch.Tensor
    cache2: torch.Tensor
    output: torch.Tensor
    policy: MarlinLaunchPolicy


def make_weights(count: int, device: torch.device):
    w13 = torch.randint(
        -(2**31), 2**31 - 1, (count, 384, 8192), dtype=torch.int32, device=device
    )
    w2 = torch.randint(
        -(2**31), 2**31 - 1, (count, 128, 12288), dtype=torch.int32, device=device
    )
    w13_scale = torch.rand(
        (count, HIDDEN // GROUP_SIZE, 4096), dtype=torch.bfloat16, device=device
    )
    w2_scale = torch.rand(
        (count, INTERMEDIATE // GROUP_SIZE, HIDDEN), dtype=torch.bfloat16, device=device
    )
    return w13, w2, w13_scale, w2_scale


def to_grace(tensors, numa_node: int):
    allocations = [
        GraceAllocation.allocate_pinned(tuple(t.shape), t.dtype, 0, numa_node)
        for t in tensors
    ]
    for allocation, source in zip(allocations, tensors, strict=True):
        allocation.copy_from(source)
    return [allocation.cuda_alias for allocation in allocations]


def make_tier(
    weights,
    ids: list[int],
    blocks_per_sm: int,
    m: int,
    device: torch.device,
    overlap_max_tokens: int,
) -> Tier:
    expert_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
    if ids:
        expert_map[torch.tensor(ids, device=device)] = torch.arange(
            len(ids), dtype=torch.int32, device=device
        )
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return Tier(
        *weights,
        expert_map=expert_map,
        # Sized at the kernel's minimum (SMs x 4), not the tier's grid: the
        # launch policy shrinks the grid but not the workspace the GEMM asserts on.
        workspace=marlin_make_workspace_new(device, 4),
        cache13=torch.empty(m * TOPK * HIDDEN, dtype=torch.bfloat16, device=device),
        cache2=torch.empty(
            (m * TOPK, INTERMEDIATE), dtype=torch.bfloat16, device=device
        ),
        output=torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device),
        policy=MarlinLaunchPolicy(
            smem_mode=ops.MARLIN_SMEM_TIGHT,
            grid_blocks=sms * blocks_per_sm,
            max_tokens=overlap_max_tokens,
        ),
    )


def call(tier: Tier, hidden, topk_ids, topk_weights):
    return fused_marlin_moe(
        hidden_states=hidden,
        w1=tier.w13,
        w2=tier.w2,
        bias1=None,
        bias2=None,
        w1_scale=tier.w13_scale,
        w2_scale=tier.w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=scalar_types.uint4b8.id,
        global_num_experts=NUM_EXPERTS,
        activation=MoEActivation.SILU,
        expert_map=tier.expert_map,
        workspace=tier.workspace,
        intermediate_cache13=tier.cache13,
        intermediate_cache2=tier.cache2,
        output=tier.output,
        is_k_full=True,
        launch_policy=tier.policy,
    )


def graph_time_us(build, warmups: int, iterations: int) -> float:
    """Median wall time of one graph replay, in microseconds."""
    for _ in range(warmups):
        build()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        build()
    torch.cuda.synchronize()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    max_hot, max_cold = max(args.hot), max(args.cold)

    hot_weights_all = make_weights(max_hot, device)
    cold_host = make_weights(max_cold, device) if max_cold else None
    cold_weights_all = to_grace(cold_host, args.numa_node) if max_cold else None
    if cold_host is not None:
        del cold_host
        torch.cuda.empty_cache()

    rows = []
    for m in args.tokens:
        hidden = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
        topk_weights = torch.full(
            (m, TOPK), 1.0 / TOPK, dtype=torch.float32, device=device
        )
        for h in args.hot:
            for c in args.cold:
                if h + c > m * TOPK:
                    continue
                hot_ids = list(range(h))
                cold_ids = list(range(128, 128 + c))
                targets = hot_ids + cold_ids
                flat = [targets[i % len(targets)] for i in range(m * TOPK)]
                topk_ids = torch.tensor(flat, dtype=torch.int32, device=device).view(
                    m, TOPK
                )

                hot_tier = make_tier(
                    tuple(t[:h] for t in hot_weights_all),
                    hot_ids,
                    2,
                    m,
                    device,
                    args.overlap_max_tokens,
                )
                cold_tier = (
                    make_tier(
                        tuple(t[:c] for t in cold_weights_all),
                        cold_ids,
                        1,
                        m,
                        device,
                        args.overlap_max_tokens,
                    )
                    if c
                    else None
                )

                hot_us = graph_time_us(
                    lambda: call(hot_tier, hidden, topk_ids, topk_weights),
                    args.warmups,
                    args.iterations,
                )
                cold_us = (
                    graph_time_us(
                        lambda: call(cold_tier, hidden, topk_ids, topk_weights),
                        args.warmups,
                        args.iterations,
                    )
                    if c
                    else 0.0
                )

                if c:
                    side = torch.cuda.Stream()

                    def union(
                        s=side,
                        ht=hot_tier,
                        ct=cold_tier,
                        h=hidden,
                        i=topk_ids,
                        w=topk_weights,
                    ):
                        main_stream = torch.cuda.current_stream()
                        fork = torch.cuda.Event()
                        fork.record(main_stream)
                        s.wait_event(fork)
                        with torch.cuda.stream(s):
                            call(ct, h, i, w)
                            join = torch.cuda.Event()
                            join.record(s)
                        call(ht, h, i, w)
                        main_stream.wait_event(join)

                    union_us = graph_time_us(union, args.warmups, args.iterations)
                else:
                    union_us = hot_us

                rows.append(
                    {
                        "tokens": m,
                        "hot_experts": h,
                        "cold_experts": c,
                        "hot_us": hot_us,
                        "cold_us": cold_us,
                        "union_us": union_us,
                        "serial_us": hot_us + cold_us,
                        "max_model_us": max(hot_us, cold_us),
                        "hot_us_per_expert": hot_us / h if h else 0.0,
                        "cold_us_per_expert": cold_us / c if c else 0.0,
                        "overlap_efficiency": (
                            (hot_us + cold_us - union_us)
                            / max(min(hot_us, cold_us), 1e-9)
                        ),
                    }
                )
                print(
                    f"m={m:3d} H={h:3d} C={c:3d}  hot {hot_us:8.1f}  cold {cold_us:8.1f}"
                    f"  union {union_us:8.1f}  max {max(hot_us, cold_us):8.1f}"
                    f"  serial {hot_us + cold_us:8.1f}",
                    flush=True,
                )
                del hot_tier, cold_tier
                torch.cuda.empty_cache()

    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.output} ({len(rows)} points)")


if __name__ == "__main__":
    main()
