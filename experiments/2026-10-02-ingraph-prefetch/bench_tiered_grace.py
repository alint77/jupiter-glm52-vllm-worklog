"""The served tiered MoE as integrated: hot tier in HBM, cold tier in pinned
Grace memory (no slot staged), real GLM-5.3 routing. Marlin as served (one
fused_marlin_moe per tier, back to back, summed) vs tiered_prefill_moe, at the
token counts the integration hands the new kernel: multi-request verify steps
(16-64) through prefill chunks. CUDA graphs, CUDA events over 20 replays."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_routing  # noqa: E402
from bench_moe_real import H, I, Q, comps, dev, mws, replay_us  # noqa: E402
from vllm.model_executor.layers.fused_moe import tiered_prefill
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.offloader.grace import GraceAllocation
from vllm.platforms import current_platform

hot, cold_hbm = comps(40), comps(24)
node = current_platform.get_device_numa_node(0)
keep, cold = [], {}
for name, t in cold_hbm.items():
    a = GraceAllocation.allocate_pinned(tuple(t.shape), t.dtype, 0, node)
    a.copy_from(t.cpu())
    keep.append(a)
    cold[name] = a.cuda_alias
k_exp = tiered_prefill.prefill_scale_exponent(hot, cold)


def marlin(x, tier, ids, wts, emap):
    return fused_marlin_moe(
        x, tier["w13_weight_packed"], tier["w2_weight_packed"], None, None,
        tier["w13_weight_scale"], tier["w2_weight_scale"], wts, ids, Q.id,
        global_num_experts=256, expert_map=emap, workspace=mws, is_k_full=True)


for chunk in (16, 32, 64, 128, 256, 512, 1024):
    rows = []
    for smp in real_routing.samples(chunk, 8, seed=chunk):
        ids = torch.from_numpy(smp.topk_ids).to(dev)
        wts = torch.rand(ids.shape, device=dev).softmax(-1)
        x = torch.randn((chunk, H), dtype=torch.bfloat16, device=dev)
        local = np.flatnonzero(smp.local_map >= 0)
        hmap = torch.full((256,), -1, dtype=torch.int32)
        cmap = torch.full((256,), -1, dtype=torch.int32)
        hmap[torch.from_numpy(local[:40])] = torch.arange(40, dtype=torch.int32)
        cmap[torch.from_numpy(local[40:])] = torch.arange(24, dtype=torch.int32)
        hmap, cmap = hmap.to(dev), cmap.to(dev)
        m = replay_us(lambda: marlin(x, hot, ids, wts, hmap)
                      + marlin(x, cold, ids, wts, cmap))
        n = replay_us(lambda: tiered_prefill.tiered_prefill_moe(
            x, ids, wts, hmap, cmap, hot, cold, k_exp))
        rows.append((m, n))
    a = np.array(rows)
    print(f"tokens {chunk:5d}: Marlin hot+Grace-cold {a[:, 0].mean():7.1f} us | "
          f"prefill kernel {a[:, 1].mean():7.1f} us ({100 * (a[:, 1].mean() / a[:, 0].mean() - 1):+.0f}%)",
          flush=True)
