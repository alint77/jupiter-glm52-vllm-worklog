#!/usr/bin/env python3
"""The integrated tiered decode path (vLLM extension), one (hot, cold) cell, for ncu.

Builds a hot tier in HBM and a cold tier on pinned Grace in Marlin's MXFP4
layout (vLLM's own repack), routes 8 tokens x top-8 with the MiMo decode token
mix, and calls `tiered_decode_moe` inside cudaProfilerStart/Stop, so

    ncu --clock-control base --profile-from-start off --csv \\
        --metrics gpu__time_duration.sum python bench_vllm_decode.py --hot 9 --cold 2

times every kernel of `--reps` layer calls at the pinned base clock. Run
NUMA-bound; the weights are random (speed does not depend on values).
"""

import argparse
import types

import torch

HIDDEN, INTER, TOPK, TOKENS, GLOBAL = 6144, 2048, 8, 8, 384
TOK_P = [0.741, 0.170, 0.054, 0.021, 0.008, 0.004, 0.001, 0.001]


def tier(count: int, device, cold: bool, numa: int):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )
    from vllm.model_executor.offloader.grace import GraceAllocation

    w13 = torch.randint(0, 256, (count, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (count, HIDDEN, INTER // 2), dtype=torch.uint8, device=device)
    s13 = torch.randint(115, 124, (count, 2 * INTER, HIDDEN // 32), dtype=torch.uint8, device=device)
    s2 = torch.randint(115, 124, (count, HIDDEN, INTER // 32), dtype=torch.uint8, device=device)
    layer = types.SimpleNamespace(params_dtype=torch.bfloat16)
    parts = prepare_moe_mxfp4_layer_for_marlin(layer, w13, w2, s13, s2, None, None)[:4]
    names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    out, keep = {}, []
    for name, t in zip(names, parts):
        if cold:
            a = GraceAllocation.allocate_pinned(tuple(t.shape), t.dtype, device.index or 0, numa)
            a.copy_from(t.cpu())
            keep.append(a)
            out[name] = a.cuda_alias
        else:
            out[name] = t
    torch.cuda.empty_cache()
    return out, keep


def routing(n_hot: int, n_cold: int, pool_hot: int, pool_cold: int, rep: int, gen, device):
    """topk_ids, weights and slot maps with n_hot + n_cold local experts active."""
    hot_map = torch.full((GLOBAL,), -1, dtype=torch.int32)
    cold_map = torch.full((GLOBAL,), -1, dtype=torch.int32)
    active = []
    for i in range(n_hot):
        hot_map[i] = (rep * n_hot + i) % pool_hot
        active.append(i)
    for i in range(n_cold):
        cold_map[100 + i] = (rep * n_cold + i) % pool_cold
        active.append(100 + i)
    ids = torch.full((TOKENS, TOPK), -1, dtype=torch.int32)
    free = [(t, k) for t in range(TOKENS) for k in range(TOPK)]
    free = [free[i] for i in torch.randperm(len(free), generator=gen).tolist()]
    used: dict[int, set[int]] = {}
    for e in active:
        n = 1 + int(torch.multinomial(torch.tensor(TOK_P), 1, generator=gen))
        for slot in list(free):
            if n == 0:
                break
            if slot[0] in used.setdefault(e, set()):
                continue
            ids[slot] = e
            used[e].add(slot[0])
            free.remove(slot)
            n -= 1
    remote = 200
    for slot in free:
        ids[slot] = remote
        remote += 1
    weights = torch.rand((TOKENS, TOPK), generator=gen)
    return ids.to(device), weights.to(device), hot_map.to(device), cold_map.to(device)


def main() -> None:
    from vllm.model_executor.layers.fused_moe.tiered_decode import tiered_decode_moe

    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", type=int, required=True)
    ap.add_argument("--cold", type=int, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--numa-node", type=int, default=1)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(0)
    pool_hot, pool_cold = max(16, args.hot), max(8, args.cold)
    hot, _ = tier(pool_hot, device, False, args.numa_node)
    cold, _keep = tier(pool_cold, device, True, args.numa_node)
    x = (torch.randn((TOKENS, HIDDEN), generator=gen) * 0.3).to(torch.bfloat16).to(device)
    calls = [routing(args.hot, args.cold, pool_hot, pool_cold, r, gen, device) for r in range(args.reps + 2)]
    for c in calls[:2]:
        tiered_decode_moe(x, c[0], c[1], c[2], c[3], hot, cold)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for c in calls[2:]:
        tiered_decode_moe(x, c[0], c[1], c[2], c[3], hot, cold)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
