"""Kernel timeline of one tiered_prefill_moe call (inside a CUDA graph) on a
real chunk: each kernel's start, duration, and the wall span per stage."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_routing  # noqa: E402
import bench_moe_real as b  # noqa: E402  (weights, helpers)
from vllm.model_executor.layers.fused_moe import tiered_prefill

for chunk in [int(a) for a in sys.argv[1:]] or (512, 2048):
    smp = real_routing.samples(chunk, 2, seed=chunk)[1]
    ids = torch.from_numpy(smp.topk_ids).to(b.dev)
    wts = torch.rand(ids.shape, device=b.dev).softmax(-1)
    x = torch.randn((chunk, b.H), dtype=torch.bfloat16, device=b.dev)
    local = np.flatnonzero(smp.local_map >= 0)
    hmap = torch.full((256,), -1, dtype=torch.int32)
    cmap = torch.full((256,), -1, dtype=torch.int32)
    hmap[torch.from_numpy(local[:40])] = torch.arange(40, dtype=torch.int32)
    cmap[torch.from_numpy(local[40:])] = torch.arange(24, dtype=torch.int32)
    hmap, cmap = hmap.to(b.dev), cmap.to(b.dev)
    fn = lambda: tiered_prefill.tiered_prefill_moe(x, ids, wts, hmap, cmap, b.hot, b.cold, b.k_exp, 0)
    for _ in range(2):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay()
        torch.accelerator.synchronize()
    ks = sorted((e for e in prof.events() if e.device_type.name == "CUDA"),
                key=lambda e: e.time_range.start)
    t0 = ks[0].time_range.start
    print(f"chunk {chunk}: routes here {int(smp.counts().sum())}, used {(smp.counts() > 0).sum()}")
    for e in ks:
        nm = e.name.split("(")[0].replace("void tiered_prefill::", "")[:40]
        print(f"   {nm:40s} start {e.time_range.start - t0:7.1f}  dur {e.device_time:7.1f}")
    print(f"   total {max(e.time_range.end for e in ks) - t0:.1f} us")
