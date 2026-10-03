"""Prefill MoE kernel iteration benchmark (run per VLLM_TIERED_PREFILL_DEFINES).

1. The 128-token tile in isolation: dense() over 64 experts x 128 tokens for
   w13 (6144 -> 4096) and w2 (2048 -> 6144), in mode 0 (full), 2 (compute only:
   stage 0 resident, no loads) and 3 (compute only, no INT4 decode). TFLOPS
   against the 630 TFLOPS reference.
2. The whole routed MoE on real GLM-5.3 routing (40 hot + 24 cold, HBM) at
   2048 and 4096 tokens, schedule 0, 4 chunks each.
Kernel time from the torch profiler over CUDA-graph replays.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vllm.model_executor.layers.fused_moe import tiered_prefill


def graph_us(fn, reps=10, match="gemm_kernel"):
    for _ in range(2):
        fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.accelerator.synchronize()
    ks = [e for e in prof.events() if e.device_type.name == "CUDA"]
    sel = sum(e.device_time for e in ks if match is None or match in e.name)
    return sel / reps


dev = torch.device("cuda", 0)
print("defines:", os.environ.get("VLLM_TIERED_PREFILL_DEFINES", "") or "(default)")
E, N = 64, 128
for k, f, name in [] if os.environ.get("BENCH_SKIP_TILE") else ((6144, 4096, "w13"), (2048, 6144, "w2")):
    q = torch.randint(-2**31, 2**31 - 1, (E, k // 16, f * 2), dtype=torch.int32, device=dev)
    s = ((0.5 + torch.rand((E, k // 32, f), device=dev) / 2) / 64).to(torch.bfloat16)
    k_exp = tiered_prefill.scale_exponent(s)
    x = torch.randn((N, k), dtype=torch.bfloat16, device=dev)
    flops = 2 * E * N * k * f
    row = []
    for mode in (0, 2, 3):
        t = graph_us(lambda: tiered_prefill.dense(x, q, s, mode, k_exp))
        row.append(f"mode{mode} {t:7.1f} us {flops / t / 1e6:5.0f} TF ({100 * flops / t / 1e6 / 630:3.0f}%)")
    print(f"{name} N={N}: " + " | ".join(row), flush=True)

import real_routing  # noqa: E402
import bench_moe_real as b  # noqa: E402

SCHEDULES = [int(v) for v in os.environ.get("BENCH_SCHEDULES", "0").split()]
for chunk, sch in [(c, s) for c in (2048, 4096) for s in SCHEDULES]:
    ts = []
    for smp in real_routing.samples(chunk, 4, seed=chunk):
        ids = torch.from_numpy(smp.topk_ids).to(dev)
        wts = torch.rand(ids.shape, device=dev).softmax(-1)
        x = torch.randn((chunk, b.H), dtype=torch.bfloat16, device=dev)
        local = np.flatnonzero(smp.local_map >= 0)
        hm = torch.full((256,), -1, dtype=torch.int32)
        cm = torch.full((256,), -1, dtype=torch.int32)
        hm[torch.from_numpy(local[:40])] = torch.arange(40, dtype=torch.int32)
        cm[torch.from_numpy(local[40:])] = torch.arange(24, dtype=torch.int32)
        hm, cm = hm.to(dev), cm.to(dev)
        # wall time: the chained widths overlap (PDL), so summing kernel
        # durations would count the overlap twice
        ts.append(b.replay_us(lambda: tiered_prefill.tiered_prefill_moe(
            x, ids, wts, hm, cm, b.hot, b.cold, b.k_exp, sch)))
    routes = chunk * 8 / 4
    floor = routes * 2 * (6144 * 4096 + 2048 * 6144) / 630e12 * 1e6
    print(f"whole MoE {chunk} tokens, schedule {sch}: {np.mean(ts):7.1f} us (compute floor {floor:.0f} us, {np.mean(ts) / floor:.2f}x)", flush=True)
