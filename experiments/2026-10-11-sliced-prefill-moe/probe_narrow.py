"""Narrow-class w13 CTA time: dense() at N tokens over E experts (4 CTAs
each, w13 6144 -> 1024), full / loads only / compute only, profiler kernel
time. Few experts = the latency-bound tail at T=4096."""
import torch
from torch.profiler import ProfilerActivity, profile

import bench_sliced as b
from vllm.model_executor.layers.fused_moe import tiered_prefill

for E in (8, 40, 132):
    w = b.comps(E, 0)
    for n in (16, 32, 64):
        x = torch.randn((n, b.H), dtype=torch.bfloat16, device=b.dev)
        row = []
        for mode in (0, 1, 2):
            f = lambda: tiered_prefill.dense(x, w["w13_weight_packed"], w["w13_weight_scale"], mode)
            for _ in range(3):
                f()
            torch.accelerator.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    f()
                torch.accelerator.synchronize()
            t = [e.device_time for e in prof.events() if e.device_type.name == "CUDA" and "gemm_kernel" in e.name]
            row.append(sum(t) / len(t))
        print(f"E={E:4d} ({4 * E} CTAs) N={n:3d}: full {row[0]:7.1f} us  loads {row[1]:7.1f}  compute {row[2]:7.1f}", flush=True)
