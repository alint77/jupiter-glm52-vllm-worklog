"""Marlin MoE (one 64-expert call, as all-HBM prefill could run it) on real
GLM-5.3 routing chunks: w13 and w2 Marlin kernel times averaged over samples,
against the real floors -- bytes of experts that got tokens, FLOPs of the
routes -- at 3.6 TB/s and 630 TFLOPS."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_routing  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types

H, I, GROUP, E = 6144, 2048, 32, 64
QUANT = scalar_types.uint4b8
W13 = 2 * I * H // 2 + 2 * I * (H // GROUP) * 2
W2 = H * I // 2 + H * (I // GROUP) * 2
F13, F2 = 2 * H * 2 * I, 2 * I * H
dev = torch.device("cuda", 0)


def weights(k, n):
    w = torch.randn((k, n), dtype=torch.bfloat16, device=dev) / k**0.5
    _, q, s, _, _, _ = marlin_quantize(w, QUANT, GROUP, act_order=False)
    return (q[None].repeat(E, 1, 1).contiguous(), s[None].repeat(E, 1, 1).contiguous())


w1, s1 = weights(H, 2 * I)
w2, s2 = weights(I, H)
ws = marlin_make_workspace_new(dev, 4)
n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 8
for chunk in (512, 1024, 2048, 4096):
    t13, t2, f13, f2 = [], [], [], []
    for smp in real_routing.samples(chunk, n_samples, seed=chunk):
        ids = torch.from_numpy(smp.topk_ids).to(dev)
        wts = torch.rand(ids.shape, device=dev).softmax(-1)
        emap = torch.from_numpy(smp.local_map).to(dev)
        x = torch.randn((chunk, H), dtype=torch.bfloat16, device=dev)

        def call():
            return fused_marlin_moe(x, w1, w2, None, None, s1, s2, wts, ids,
                                    QUANT.id, global_num_experts=256,
                                    expert_map=emap, workspace=ws, is_k_full=True)

        for _ in range(2):
            call()
        torch.accelerator.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call()
        g.replay()
        torch.accelerator.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                g.replay()
            torch.accelerator.synchronize()
        m = [e.device_time for e in prof.events()
             if e.device_type.name == "CUDA" and "Marlin" in e.name]
        t13.append(np.mean(m[0::2]))
        t2.append(np.mean(m[1::2]))
        c = smp.counts()
        f13.append((int((c > 0).sum()), int(c.sum())))
    used = np.mean([u for u, _ in f13])
    routes = np.mean([r for _, r in f13])
    mem13, mem2 = used * W13 / 3.6e6, used * W2 / 3.6e6
    cmp13, cmp2 = routes * F13 / 630e6, routes * F2 / 630e6
    print(f"chunk {chunk}: {n_samples} samples, {used:.1f}/64 experts used, "
          f"{routes:.0f} routes/GPU | Marlin w13 {np.mean(t13):6.1f} us "
          f"(floor {max(mem13, cmp13):5.1f}: mem {mem13:5.1f}, compute {cmp13:5.1f}) "
          f"| w2 {np.mean(t2):6.1f} us (floor {max(mem2, cmp2):5.1f})")
