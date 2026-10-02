"""Real-routing w13 on one GPU's 64 experts: Marlin's GEMM kernel vs the
grouped wgmma kernel's launches (one per tile width, widest first), same
chunks. Also the cost of sorting/gathering activation rows (torch ops here;
a fused kernel later). Graph + profiler, mean over samples."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_routing  # noqa: E402
from vllm.model_executor.layers.fused_moe import tiered_prefill
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
from vllm.scalar_type import scalar_types

H, I, E, QUANT = 6144, 2048, 64, scalar_types.uint4b8
W13 = 2 * I * H // 2 + 2 * I * (H // 32) * 2
dev = torch.device("cuda", 0)
ext = tiered_prefill._extension()


def weights(k, n):
    w = torch.randn((k, n), dtype=torch.bfloat16, device=dev) / k**0.5
    _, q, s, _, _, _ = marlin_quantize(w, QUANT, 32, act_order=False)
    return q[None].repeat(E, 1, 1).contiguous(), s[None].repeat(E, 1, 1).contiguous()


def timed(fn, key):
    for _ in range(2):
        fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            g.replay()
        torch.accelerator.synchronize()
    ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
    out = {}
    for k in key:
        if k == "span":  # wall span of our launches per replay (they overlap)
            g_ev = sorted((e for e in ev if "gemm_kernel" in e.name),
                          key=lambda e: e.time_range.start)
            # replays run back to back: first start to last end, per replay
            out[k] = (max(e.time_range.end for e in g_ev)
                      - g_ev[0].time_range.start) / 10
        else:
            out[k] = sum(e.device_time for e in ev if k_match(e.name, k)) / 10
    return out


def k_match(name, k):
    return {"marlin": "Marlin" in name, "ours": "gemm_kernel" in name,
            "all": True}[k]


w1, s1 = weights(H, 2 * I)
w2, s2 = weights(I, H)
ws = marlin_make_workspace_new(dev, 4)
k_exp = tiered_prefill.scale_exponent(s1)
n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 16
for chunk in (512, 1024, 2048, 4096):
    rows = []
    for smp in real_routing.samples(chunk, n_samples, seed=chunk):
        ids = torch.from_numpy(smp.topk_ids).to(dev)
        lmap = torch.from_numpy(smp.local_map).to(dev)
        wts = torch.rand(ids.shape, device=dev).softmax(-1)
        x = torch.randn((chunk, H), dtype=torch.bfloat16, device=dev)
        # Marlin: its w13 kernel inside the full call
        mar = timed(lambda: fused_marlin_moe(
            x, w1, w2, None, None, s1, s2, wts, ids, QUANT.id,
            global_num_experts=256, expert_map=lmap, workspace=ws,
            is_k_full=True), ["marlin"])["marlin"]
        # ours: host-built tables, then only the launches are timed
        loc = lmap[ids.long()].reshape(-1)
        route = torch.nonzero(loc >= 0).squeeze(1)
        route = route[torch.argsort(loc[route], stable=True)]
        expert = loc[route].long()
        token = route // 8
        xh = torch.empty((chunk, H), dtype=torch.float16, device=dev)
        xs = torch.empty((chunk,), dtype=torch.float32, device=dev)
        ext.prep(xh, xs, x)
        xh_r, xs_r = xh[token].contiguous(), xs[token].contiguous()
        y = torch.empty((route.numel(), 2 * I), dtype=torch.bfloat16, device=dev)
        counts = torch.bincount(expert, minlength=E).tolist()
        by_nt, row = {}, 0
        for e, n in enumerate(counts):
            for r0 in range(0, n, 128):
                r = min(128, n - r0)
                nt = next(c for c in (16, 32, 48, 64, 96, 128) if c >= r)
                by_nt.setdefault(nt, []).append((e, 0, row + r0, r))
            row += n
        tabs = [(nt, torch.tensor(by_nt[nt], dtype=torch.int32, device=dev))
                for nt in sorted(by_nt, reverse=True)]

        if not tabs:  # no routes on this GPU
            continue

        def ours():
            for i, (nt, u) in enumerate(tabs):
                ext.grouped(y, xh_r, xs_r, w1, s1, u, nt, k_exp, 0, i > 0)

        o = timed(ours, ["span"])["span"]
        tok = token

        def gather():
            ext.prep(xh, xs, x)
            return xh[tok].contiguous(), xs[tok].contiguous()

        gat = timed(gather, ["all"])["all"]
        c = np.array(counts)
        rows.append((mar, o, gat, (c > 0).sum(), c.sum(), c.max()))
    a = np.array(rows, dtype=float)
    med = np.median(a[:, 1] / a[:, 0])
    floor = a[:, 3].mean() * W13 / 3.6e6
    print(f"chunk {chunk}: experts used {a[:, 3].mean():.0f}/64, routes {a[:, 4].mean():.0f}, "
          f"busiest {a[:, 5].mean():.0f} | Marlin w13 {a[:, 0].mean():6.1f} us | ours "
          f"{a[:, 1].mean():6.1f} us ({100 * (a[:, 1].mean() / a[:, 0].mean() - 1):+.0f}%) "
          f"+ prep/gather {a[:, 2].mean():5.1f} | weight-read floor {floor:5.1f} | "
          f"median ours/Marlin {med:.2f}")
