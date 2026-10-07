"""Every bf16 linear of the GLM-5.3 verify step at M=8 (TP4 shards): F.linear
vs skinny_v6 configs, L2 flushed before each launch (torch profiler, median of
50), max error against an fp32 reference.

    bench_shapes.py
"""
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
ext = load(name="skinny_v6", sources=[str(HERE / "skinny_v6.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
           build_directory="/e/fscratch/profound/naeimitabiei1/caches/skinny_v6")
dev = torch.device("cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)


def timed(fn, reps=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            flush.fill_(1)
            fn()
        torch.cuda.synchronize()
    ds = [e.duration_ns() / 1e3 for e in p.profiler.kineto_results.events()
          if e.device_type().name == "CUDA" and "Fill" not in e.name()]
    per = len(ds) // reps
    return st.median(sum(ds[i * per:(i + 1) * per]) for i in range(reps))


SHAPES = {  # name: (N, K, calls per verify step)
    "o_proj": (6144, 4096, 78), "fused_qkv_a": (2624, 6144, 78), "q_b": (4096, 2048, 78),
    "indexer wq_b": (4096, 2048, 21), "indexer wk": (160, 6144, 21),
    "shared gate_up": (1024, 6144, 75), "shared down": (6144, 512, 75),
    "dense gate_up": (6144, 6144, 3), "dense down": (6144, 3072, 3),
}
CFGS = ((2, 4), (2, 8), (4, 2), (4, 4), (4, 8), (8, 2), (8, 4))
tot_c = tot_s = 0.0
for name, (n, k, calls) in SHAPES.items():
    w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
    x = torch.randn(8, k, device=dev).to(torch.bfloat16)
    ref = x.float() @ w.float().t()
    e_cub = (F.linear(x, w).float() - ref).abs().max().item()
    t_cub = timed(lambda: F.linear(x, w))
    best = (1e9, None)
    for wp, un in CFGS:
        if k % (32 * wp * un) or n % 16:
            continue
        err = (ext.gemm(x, w, wp, un, False).float() - ref).abs().max().item()
        assert err <= 2 * e_cub + 1e-2, (name, wp, un, err)
        best = min(best, (timed(lambda: ext.gemm(x, w, wp, un, False)), (wp, un)))
    gain = t_cub - best[0]
    tot_c += calls * t_cub
    tot_s += calls * min(t_cub, best[0])
    print(f"{name:15s} N {n:5d} K {k:5d} x{calls:2d}  cuBLAS {t_cub:6.2f}  best v6 {best[0]:6.2f} "
          f"{best[1]}  gain {gain:+6.2f} us/call  {calls * gain / 1e3:+.3f} ms/step")
print(f"per verify step: cuBLAS {tot_c / 1e3:.2f} ms, best-of {tot_s / 1e3:.2f} ms "
      f"(-{(tot_c - tot_s) / 1e3:.2f} ms)")
