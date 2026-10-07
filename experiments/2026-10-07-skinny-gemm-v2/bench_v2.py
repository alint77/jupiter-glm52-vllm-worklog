"""skinny_v2 / v3 against F.linear on the decode GEMM shapes (M=8): error against an
fp32 reference, and kernel time with L2 flushed before each launch (torch
profiler, median of 50).

    bench_v2.py
"""
import itertools
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
build = Path("/e/fscratch/profound/naeimitabiei1/caches/skinny_v2")
build.mkdir(parents=True, exist_ok=True)
ext = load(name="skinny_v2", sources=[str(HERE / "skinny_v2.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
           build_directory=str(build))
build4 = Path("/e/fscratch/profound/naeimitabiei1/caches/skinny_v4")
build4.mkdir(parents=True, exist_ok=True)
ext4 = load(name="skinny_v4", sources=[str(HERE / "skinny_v4.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            build_directory=str(build4))
build5 = Path("/e/fscratch/profound/naeimitabiei1/caches/skinny_v5")
build5.mkdir(parents=True, exist_ok=True)
ext5 = load(name="skinny_v5", sources=[str(HERE / "skinny_v5.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            build_directory=str(build5))
build3 = Path("/e/fscratch/profound/naeimitabiei1/caches/skinny_v3")
build3.mkdir(parents=True, exist_ok=True)
ext3 = load(name="skinny_v3", sources=[str(HERE / "skinny_v3.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            build_directory=str(build3))
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


SHAPES = {"o_proj": (6144, 4096), "fused_qkv_a": (2624, 6144), "q_b": (4096, 2048)}
for name, (n, k) in SHAPES.items():
    w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
    x = torch.randn(8, k, device=dev).to(torch.bfloat16)
    ref = x.float() @ w.float().t()
    base = F.linear(x, w)
    e_cublas = (base.float() - ref).abs().max().item()
    res = {"F.linear": timed(lambda: F.linear(x, w))}
    for wp, un in itertools.product((4, 8, 16), (2, 4, 8)):
        if k % (32 * wp * un) or (wp, un) in ((16, 8),):
            continue
        y = ext.gemm(x, w, wp, un)
        err = (y.float() - ref).abs().max().item()
        assert err <= 2 * e_cublas + 1e-2, (name, wp, un, err, e_cublas)
        res[f"v2 w{wp} u{un}"] = timed(lambda: ext.gemm(x, w, wp, un))
    for wp, un in ((2, 4), (2, 8), (4, 2), (4, 4), (4, 8), (8, 2), (8, 4), (16, 2)):
        if k % (32 * wp * un):
            continue
        y = ext4.gemm(x, w, wp, un)
        err = (y.float() - ref).abs().max().item()
        assert err <= 2 * e_cublas + 1e-2, (name, wp, un, err)
        res[f"v4 w{wp} u{un}"] = timed(lambda: ext4.gemm(x, w, wp, un))
    part = torch.empty(n // 16 * 8 * 128, dtype=torch.float32, device=dev)
    count = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    for (wp, un), ks in itertools.product(((2, 4), (2, 8), (4, 2), (4, 4), (4, 8), (8, 2), (8, 4)), (1, 2, 3, 4)):
        if k % (32 * wp * un * ks):
            continue
        y = ext5.gemm(x, w, part, count, wp, un, ks)
        y2 = ext5.gemm(x, w, part, count, wp, un, ks)
        err = (y.float() - ref).abs().max().item()
        assert err <= 2 * e_cublas + 1e-2 and torch.equal(y, y2), (name, wp, un, ks, err)
        res[f"v5 w{wp} u{un} k{ks}"] = timed(lambda: ext5.gemm(x, w, part, count, wp, un, ks))
    mb = n * k * 2 / 1e6
    print(f"\n{name}: {mb:.1f} MB, cuBLAS max err {e_cublas:.2e}")
    for key, us in sorted(res.items(), key=lambda x: x[1])[:6]:
        print(f"  {key:16s} {us:6.2f} us  {mb / us:5.2f} TB/s")
    print(f"  F.linear     {res['F.linear']:6.2f} us")
