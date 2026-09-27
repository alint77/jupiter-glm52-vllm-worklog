#!/usr/bin/env python3
"""Production Marlin MXFP4 MoE on MiMo shapes, one tier at a time, for ncu.

Runs `fused_marlin_moe` exactly as the tiered path launches it (tight smem,
hot grid SMs x 2, cold grid SMs x 1, block_m 16 max tokens) for one tier with
`count` active experts, 8 tokens x top-8 routing, and per-expert token counts
drawn from the MiMo decode mix. Weights sit in HBM (hot) or pinned Grace over
UVA (cold). The measured calls are wrapped in cudaProfilerStart/Stop so

    ncu --clock-control base --profile-from-start off --csv \\
        --metrics gpu__time_duration.sum python marlin_mimo.py --tier cold --count 2

times every kernel of `--reps` calls at the pinned base clock. Run NUMA-bound.
"""

import argparse
import types

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
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.model_executor.offloader.grace import GraceAllocation
from vllm.scalar_type import scalar_types

HIDDEN, INTER, TOPK, TOKENS, GLOBAL_EXPERTS = 6144, 2048, 8, 8, 384
# MiMo-V2.6 expert e8m0 exponents 111..124 (60 sampled tensors)
SCALE_P = [6.61e-06, 3.59e-04, 3.13e-03, 1.43e-02, 3.28e-02, 7.83e-02, 5.18e-02,
           1.49e-02, 1.37e-02, 6.70e-01, 1.20e-01, 4.26e-04, 1.55e-05, 5.51e-07]
TOK_P = [0.741, 0.170, 0.054, 0.021, 0.008, 0.004, 0.001, 0.001]


def scales(shape, gen):
    p = torch.tensor(SCALE_P)
    idx = torch.multinomial(p, int(torch.tensor(shape).prod()), replacement=True, generator=gen)
    return (idx + 111).to(torch.uint8).view(shape)


def make_pool(count: int, device, gen):
    w13 = torch.randint(0, 256, (count, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, generator=gen)
    w2 = torch.randint(0, 256, (count, HIDDEN, INTER // 2), dtype=torch.uint8, generator=gen)
    s13 = scales((count, 2 * INTER, HIDDEN // 32), gen)
    s2 = scales((count, HIDDEN, INTER // 32), gen)
    layer = types.SimpleNamespace(params_dtype=torch.bfloat16)
    w13, w2, s13, s2, _, _ = prepare_moe_mxfp4_layer_for_marlin(
        layer, w13.to(device), w2.to(device), s13.to(device), s2.to(device), None, None)
    return w13, w2, s13, s2


def to_grace(tensors, numa_node):
    out = []
    for t in tensors:
        a = GraceAllocation.allocate_pinned(tuple(t.shape), t.dtype, 0, numa_node)
        a.copy_from(t)
        out.append(a.cuda_alias)
    return out


def routing(count: int, first: int, gen, device):
    """topk_ids with `count` local experts (global ids first..first+count-1)
    active, token counts from the decode mix; other slots go to remote ids."""
    ids = torch.full((TOKENS, TOPK), -1, dtype=torch.int64)
    free = [(t, k) for t in range(TOKENS) for k in range(TOPK)]
    order = torch.randperm(len(free), generator=gen).tolist()
    free = [free[i] for i in order]
    used_tok: dict[int, set[int]] = {}
    for e in range(count):
        n = 1 + int(torch.multinomial(torch.tensor(TOK_P), 1, generator=gen))
        placed = 0
        for slot in list(free):
            t, _ = slot
            if placed == n:
                break
            if t in used_tok.setdefault(e, set()):
                continue
            ids[slot] = first + e
            used_tok[e].add(t)
            free.remove(slot)
            placed += 1
    remote = 200
    for slot in free:  # distinct remote experts per token
        ids[slot] = remote
        remote += 1
    weights = torch.rand((TOKENS, TOPK), generator=gen)
    return ids.to(device=device, dtype=torch.int32), weights.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["hot", "cold"], required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--numa-node", type=int, default=1)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(0)
    pool = max(args.pool, args.count)
    weights = make_pool(pool, device, gen)
    if args.tier == "cold":
        weights = to_grace(weights, args.numa_node)
        torch.cuda.empty_cache()
    w13, w2, s13, s2 = weights
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    policy = MarlinLaunchPolicy(smem_mode=ops.MARLIN_SMEM_TIGHT,
                                grid_blocks=sms * (2 if args.tier == "hot" else 1),
                                max_tokens=16)
    workspace = marlin_make_workspace_new(device, 4)
    cache13 = torch.empty(TOKENS * TOPK * max(2 * INTER, HIDDEN), dtype=torch.bfloat16, device=device)
    cache2 = torch.empty((TOKENS * TOPK, INTER), dtype=torch.bfloat16, device=device)
    out = torch.empty((TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    x = torch.randn((TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)

    calls = []
    for r in range(args.reps + 2):
        expert_map = torch.full((GLOBAL_EXPERTS,), -1, dtype=torch.int32, device=device)
        local = [(r * args.count + e) % pool for e in range(args.count)]
        expert_map[:args.count] = torch.tensor(local, dtype=torch.int32, device=device)
        calls.append((expert_map, *routing(args.count, 0, gen, device)))

    def run(expert_map, topk_ids, topk_weights):
        fused_marlin_moe(
            hidden_states=x, w1=w13, w2=w2, bias1=None, bias2=None,
            w1_scale=s13, w2_scale=s2, topk_weights=topk_weights, topk_ids=topk_ids,
            quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=GLOBAL_EXPERTS,
            activation=MoEActivation.SILU, expert_map=expert_map, workspace=workspace,
            intermediate_cache13=cache13, intermediate_cache2=cache2, output=out,
            is_k_full=True, launch_policy=policy)

    for c in calls[:2]:
        run(*c)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for c in calls[2:]:
        run(*c)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
