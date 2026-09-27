#!/usr/bin/env python3
"""Production two-tier Marlin MoE for one (hot, cold) cell, timed by graph replay.

The wall-clock counterpart to `bench_vllm_decode.py --wall`: the same tiers,
routing and token mix, run the way `apply_tiered` launches decode today --
cold tier on a side stream (tight smem, SMs x 1), hot on the main stream
(SMs x 2), joined and summed -- captured 20 calls to a CUDA graph and replayed.
Each tier aligns its own routes here; production replaces that with one fused
replica-align kernel that the tiered path pays too, so this slightly favours
the new path.

    marlin_wall.py --hot 9 --cold 2     (run NUMA-bound)
"""

import argparse
import sys
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
from vllm.scalar_type import scalar_types

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_vllm_decode import GLOBAL, HIDDEN, INTER, TOKENS, TOPK, routing, tier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", type=int, required=True)
    ap.add_argument("--cold", type=int, required=True)
    ap.add_argument("--numa-node", type=int, default=1)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(0)
    pool_hot, pool_cold = max(16, args.hot), max(8, args.cold)
    hot, _ = tier(pool_hot, device, False, args.numa_node)
    cold, _keep = tier(pool_cold, device, True, args.numa_node)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    x = (torch.randn((TOKENS, HIDDEN), generator=gen) * 0.3).to(torch.bfloat16).to(device)
    calls = [routing(args.hot, args.cold, pool_hot, pool_cold, r, gen, device) for r in range(22)]

    def state(blocks):
        return {
            "policy": MarlinLaunchPolicy(smem_mode=ops.MARLIN_SMEM_TIGHT, grid_blocks=sms * blocks,
                                         max_tokens=16),
            "workspace": marlin_make_workspace_new(device, 4),
            "c13": torch.empty(TOKENS * TOPK * max(2 * INTER, HIDDEN), dtype=torch.bfloat16, device=device),
            "c2": torch.empty((TOKENS * TOPK, INTER), dtype=torch.bfloat16, device=device),
            "out": torch.empty((TOKENS, HIDDEN), dtype=torch.bfloat16, device=device),
        }

    st = {"hot": state(2), "cold": state(1)}
    side = torch.cuda.Stream()

    def run_tier(name, t, emap, ids, wts):
        s = st[name]
        return fused_marlin_moe(
            hidden_states=x, w1=t["w13_weight"], w2=t["w2_weight"], bias1=None, bias2=None,
            w1_scale=t["w13_weight_scale"], w2_scale=t["w2_weight_scale"],
            topk_weights=wts, topk_ids=ids, quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=GLOBAL, activation=MoEActivation.SILU, expert_map=emap,
            workspace=s["workspace"], intermediate_cache13=s["c13"], intermediate_cache2=s["c2"],
            output=s["out"], is_k_full=True, launch_policy=s["policy"])

    def layer(c):
        ids, wts, hot_map, cold_map = c
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            cold_out = run_tier("cold", cold, cold_map, ids, wts)
        hot_out = run_tier("hot", hot, hot_map, ids, wts)
        main.wait_stream(side)
        return cold_out.add_(hot_out)

    for c in calls[:2]:
        layer(c)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(20):
            layer(calls[2 + i % 20])
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(10):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            graph.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / 200)
    print(f"hot={args.hot} cold={args.cold}: {best:.1f} us per layer call (Marlin, graph replay)")


if __name__ == "__main__":
    main()
