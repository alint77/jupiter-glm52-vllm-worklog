#!/usr/bin/env python3
"""Is the tiered-MoE overlap token threshold in the right place?

`_apply_tier_launch_policy` gives the two Marlin tiers a tight shared-memory
launch policy so they can be co-resident and overlap, but only at or below
`tiered_overlap_max_tokens = (num_speculative_tokens + 1) * max_num_seqs` -- 16
in production. Above it `fused_marlin_moe` falls back to the default heuristic,
which asks for the whole SM's shared memory, so the tiers serialize.

That number is not a tuned threshold. It is "the largest step this decode
configuration can produce", so it functions as a decode-versus-prefill
discriminator. Whether serializing is actually right above it was asserted in a
comment, not measured. A mixed prefill-decode step is exactly what crosses it,
and prefill/decode co-scheduling is the top open lever in the worklog.

For each token count this measures the same work three ways:

    policy_on    tight smem, fixed grid, two streams  (what decode does)
    policy_off   default heuristic, two streams       (what >16 does today)
    serial       default heuristic, one stream        (lower bound on policy_off)

If policy_on beats policy_off above 16 tokens, the threshold is leaving
throughput on the table and should move.
"""

import argparse
import json
import statistics
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
GROUP_SIZE = 32
UNBOUNDED = 10**9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+",
        default=[4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument(
        "--hot", type=int, default=33,
        help="hot experts activated per layer per rank; 2496 slots / 75 layers",
    )
    parser.add_argument("--cold", type=int, default=31, help="the rest of the 64 owned")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_weights(count, device):
    return (
        torch.randint(-(2**31), 2**31 - 1, (count, 384, 8192), dtype=torch.int32, device=device),
        torch.randint(-(2**31), 2**31 - 1, (count, 128, 12288), dtype=torch.int32, device=device),
        torch.rand((count, HIDDEN // GROUP_SIZE, 4096), dtype=torch.bfloat16, device=device),
        torch.rand((count, INTERMEDIATE // GROUP_SIZE, HIDDEN), dtype=torch.bfloat16, device=device),
    )


def to_grace(tensors, numa_node):
    allocations = [
        GraceAllocation.allocate_pinned(tuple(t.shape), t.dtype, 0, numa_node)
        for t in tensors
    ]
    for allocation, source in zip(allocations, tensors, strict=True):
        allocation.copy_from(source)
    return [a.cuda_alias for a in allocations]


def expert_map_for(ids, device):
    m = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
    m[torch.tensor(ids, device=device)] = torch.arange(
        len(ids), dtype=torch.int32, device=device
    )
    return m


def workspaces(device, m, _blocks_per_sm):
    return {
        # The kernel's min workspace is SMs x 4 regardless of the launch grid,
        # so this must not be sized from the tier's blocks_per_sm.
        "workspace": marlin_make_workspace_new(device, 4),
        "intermediate_cache13": torch.empty(
            m * TOPK * HIDDEN, dtype=torch.bfloat16, device=device
        ),
        "intermediate_cache2": torch.empty(
            (m * TOPK, INTERMEDIATE), dtype=torch.bfloat16, device=device
        ),
        "output": torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device),
    }


def call(weights, expert_map, ws, hidden, topk_ids, topk_weights, policy):
    return fused_marlin_moe(
        hidden_states=hidden,
        w1=weights[0], w2=weights[1], bias1=None, bias2=None,
        w1_scale=weights[2], w2_scale=weights[3],
        topk_weights=topk_weights, topk_ids=topk_ids,
        quant_type_id=scalar_types.uint4b8.id,
        global_num_experts=NUM_EXPERTS,
        activation=MoEActivation.SILU,
        expert_map=expert_map, is_k_full=True,
        launch_policy=policy, **ws,
    )


def graph_time_us(build, warmups, iterations):
    for _ in range(warmups):
        build()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        build()
    torch.cuda.synchronize()
    for _ in range(3):
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
    sms = torch.cuda.get_device_properties(device).multi_processor_count

    hot_w = make_weights(args.hot, device)
    cold_host = make_weights(args.cold, device)
    cold_w = to_grace(cold_host, args.numa_node)
    del cold_host
    torch.cuda.empty_cache()

    hot_ids = list(range(args.hot))
    cold_ids = list(range(128, 128 + args.cold))
    hot_map = expert_map_for(hot_ids, device)
    cold_map = expert_map_for(cold_ids, device)
    targets = hot_ids + cold_ids

    hot_policy = MarlinLaunchPolicy(
        smem_mode=ops.MARLIN_SMEM_TIGHT, grid_blocks=sms * 2, max_tokens=UNBOUNDED
    )
    cold_policy = MarlinLaunchPolicy(
        smem_mode=ops.MARLIN_SMEM_TIGHT, grid_blocks=sms * 1, max_tokens=UNBOUNDED
    )

    rows = []
    for m in args.tokens:
        hidden = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
        topk_weights = torch.full(
            (m, TOPK), 1.0 / TOPK, dtype=torch.float32, device=device
        )
        flat = [targets[i % len(targets)] for i in range(m * TOPK)]
        topk_ids = torch.tensor(flat, dtype=torch.int32, device=device).view(m, TOPK)
        hot_ws = workspaces(device, m, 2)
        cold_ws = workspaces(device, m, 1)
        side = torch.cuda.Stream()

        def two_stream(hp, cp):
            def build():
                main_stream = torch.cuda.current_stream()
                fork = torch.cuda.Event()
                fork.record(main_stream)
                side.wait_event(fork)
                with torch.cuda.stream(side):
                    call(cold_w, cold_map, cold_ws, hidden, topk_ids, topk_weights, cp)
                    join = torch.cuda.Event()
                    join.record(side)
                call(hot_w, hot_map, hot_ws, hidden, topk_ids, topk_weights, hp)
                main_stream.wait_event(join)
            return build

        def one_stream():
            call(cold_w, cold_map, cold_ws, hidden, topk_ids, topk_weights, None)
            call(hot_w, hot_map, hot_ws, hidden, topk_ids, topk_weights, None)

        on = graph_time_us(two_stream(hot_policy, cold_policy), args.warmups, args.iterations)
        off = graph_time_us(two_stream(None, None), args.warmups, args.iterations)
        serial = graph_time_us(one_stream, args.warmups, args.iterations)

        rows.append(
            {
                "tokens": m,
                "hot_experts": args.hot,
                "cold_experts": args.cold,
                "policy_on_us": on,
                "policy_off_us": off,
                "serial_us": serial,
                "policy_gain_pct": 100.0 * (off - on) / off,
                "inside_production_threshold": m <= 16,
            }
        )
        print(
            f"m={m:5d}  policy_on {on:9.1f}  policy_off {off:9.1f}  serial {serial:9.1f}"
            f"  gain {100.0 * (off - on) / off:+6.1f}%"
            f"{'   <- production overlaps here' if m <= 16 else ''}",
            flush=True,
        )
        del hot_ws, cold_ws
        torch.cuda.empty_cache()

    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
