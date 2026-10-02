"""Whole routed-expert MoE on one GPU, real GLM-5.3 chunks: Marlin's
fused_marlin_moe (64 experts in HBM, one call) vs tiered_prefill_moe (40 hot
+ 24 cold, both in HBM). Both captured in CUDA graphs; time per replay from
CUDA events around 20 replays (a replay is one launch)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_routing  # noqa: E402
from vllm.model_executor.layers.fused_moe import tiered_prefill
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
from vllm.scalar_type import scalar_types

H, I, Q = 6144, 2048, scalar_types.uint4b8
dev = torch.device("cuda", 0)


def tier(n, k_, f):
    w = torch.randn((k_, f), dtype=torch.bfloat16, device=dev) / k_**0.5
    _, q, s, _, _, _ = marlin_quantize(w, Q, 32, False)
    return q[None].repeat(n, 1, 1).contiguous(), s[None].repeat(n, 1, 1).contiguous()


def comps(n):
    w13, s13 = tier(n, H, 2 * I)
    w2, s2 = tier(n, I, H)
    return {"w13_weight_packed": w13, "w13_weight_scale": s13,
            "w2_weight_packed": w2, "w2_weight_scale": s2}


allw, hot, cold = comps(64), comps(40), comps(24)
k_exp = tiered_prefill.prefill_scale_exponent(hot, cold)
mws = marlin_make_workspace_new(dev, 4)


def replay_us(fn):
    for _ in range(2):
        fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.accelerator.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(20):
        g.replay()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) * 1000 / 20


n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 16
for chunk in ((512, 1024, 2048, 4096) if __name__ == "__main__" else ()):
    rows = []
    for smp in real_routing.samples(chunk, n_samples, seed=chunk):
        ids = torch.from_numpy(smp.topk_ids).to(dev)
        wts = torch.rand(ids.shape, device=dev).softmax(-1)
        x = torch.randn((chunk, H), dtype=torch.bfloat16, device=dev)
        lmap = torch.from_numpy(smp.local_map).to(dev)
        local = np.flatnonzero(smp.local_map >= 0)
        hmap = torch.full((256,), -1, dtype=torch.int32)
        cmap = torch.full((256,), -1, dtype=torch.int32)
        hmap[torch.from_numpy(local[:40])] = torch.arange(40, dtype=torch.int32)
        cmap[torch.from_numpy(local[40:])] = torch.arange(24, dtype=torch.int32)
        hmap, cmap = hmap.to(dev), cmap.to(dev)
        m = replay_us(lambda: fused_marlin_moe(
            x, allw["w13_weight_packed"], allw["w2_weight_packed"], None, None,
            allw["w13_weight_scale"], allw["w2_weight_scale"], wts, ids, Q.id,
            global_num_experts=256, expert_map=lmap, workspace=mws, is_k_full=True))
        o0 = replay_us(lambda: tiered_prefill.tiered_prefill_moe(
            x, ids, wts, hmap, cmap, hot, cold, k_exp, 0))
        o = replay_us(lambda: tiered_prefill.tiered_prefill_moe(
            x, ids, wts, hmap, cmap, hot, cold, k_exp, 2))
        o3 = replay_us(lambda: tiered_prefill.tiered_prefill_moe(
            x, ids, wts, hmap, cmap, hot, cold, k_exp, 4))
        rows.append((m, o, o0, o3))
    a = np.array(rows)
    print(f"chunk {chunk}: Marlin MoE {a[:, 0].mean():7.1f} us | chained {a[:, 2].mean():7.1f} | "
          f"persistent-2 {a[:, 3].mean():7.1f} | persistent {a[:, 1].mean():7.1f} us "
          f"({100 * (a[:, 1].mean() / a[:, 0].mean() - 1):+.0f}%), median ratio "
          f"{np.median(a[:, 1] / a[:, 0]):.2f}")
