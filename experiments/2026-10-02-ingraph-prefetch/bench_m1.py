"""Milestone 1 timing: 64 experts, every expert on the same N tokens, graph +
profiler. Against the HBM floor (3.6 TB/s) and the 630 TFLOPS roof."""

import torch
from torch.profiler import ProfilerActivity, profile

import sys

from vllm.model_executor.layers.fused_moe import tiered_prefill

LOADS = 1 if "--loads-only" in sys.argv else 2 if "--compute-only" in sys.argv else 0

E, GROUP = 64, 32
dev = torch.device("cuda", 0)
for k, f, name in ((6144, 4096, "w13"), (2048, 6144, "w2")):
    q = torch.randint(-2**31, 2**31 - 1, (E, k // 16, f * 2), dtype=torch.int32, device=dev)
    s = (torch.rand((E, k // GROUP, f), device=dev) / 64).to(torch.bfloat16)
    wbytes = q.numel() * 4 + s.numel() * 2
    for n in (8, 16, 24, 32, 48, 64):
        x = torch.randn((n, k), dtype=torch.bfloat16, device=dev)
        for _ in range(3):
            tiered_prefill.dense(x, q, s, LOADS)
        torch.accelerator.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            tiered_prefill.dense(x, q, s, LOADS)
        g.replay()
        torch.accelerator.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                g.replay()
            torch.accelerator.synchronize()
        # every launch of a call (N > 32 runs in chunks), per replay
        t = sum(e.device_time for e in prof.events()
                if e.device_type.name == "CUDA" and "gemm_kernel" in e.name) / 20
        flops = 2 * E * n * k * f
        print(f"{name} N={n:3d}: {t:7.1f} us  {wbytes / t / 1e6:5.2f} TB/s "
              f"({100 * wbytes / t / 1e6 / 3.6:3.0f}% of 3.6)  "
              f"{flops / t / 1e6:6.1f} TFLOPS ({100 * flops / t / 1e6 / 630:3.0f}%)  "
              f"floor {wbytes / 3.6e6:6.1f} us")
