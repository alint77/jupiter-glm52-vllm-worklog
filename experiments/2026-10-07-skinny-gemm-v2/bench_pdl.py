"""In a CUDA graph, behind a real predecessor (a cuBLAS M=8 GEMM on a 4 MiB
weight, like W_UV before o_proj): F.linear vs skinny_v6 without / with
programmatic dependent launch. Reported: the GEMM kernel's own time and the
span from the predecessor's start to the GEMM's end (median of 100 replays,
L2 flushed before each).

    bench_pdl.py
"""
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
b6 = Path("/e/fscratch/profound/naeimitabiei1/caches/skinny_v6")
b6.mkdir(parents=True, exist_ok=True)
ext = load(name="skinny_v6", sources=[str(HERE / "skinny_v6.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], build_directory=str(b6))
dev = torch.device("cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
wp_ = torch.randn(2048, 1024, dtype=torch.bfloat16, device=dev)  # 4 MiB predecessor
xp = torch.randn(8, 1024, dtype=torch.bfloat16, device=dev)


def run(fn, reps=100):
    g = torch.cuda.CUDAGraph()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            flush.fill_(1)
            g.replay()
        torch.cuda.synchronize()
    ev = sorted(((e.start_ns(), e.duration_ns(), e.name()) for e in p.profiler.kineto_results.events()
                 if e.device_type().name == "CUDA"), key=lambda e: e[0])
    reps_, cur = [], []
    for s, d, n in ev:
        if "Fill" in n:
            if cur:
                reps_.append(cur)
            cur = []
        else:
            cur.append((s, d, n))
    reps_.append(cur)
    spans = [(r[-1][0] + r[-1][1] - r[0][0]) / 1e3 for r in reps_ if len(r) >= 2]
    gemm = [r[-1][1] / 1e3 for r in reps_ if len(r) >= 2]
    return st.median(gemm), st.median(spans)


for name, (n, k), cfg in (("o_proj", (6144, 4096), (4, 4)), ("fused_qkv_a", (2624, 6144), (4, 4)),
                          ("q_b", (4096, 2048), (4, 4))):
    w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
    x = torch.randn(8, k, device=dev).to(torch.bfloat16)
    pre = lambda: F.linear(xp, wp_)  # noqa: E731
    ref = F.linear(x, w)
    for pdl in (False, True):
        assert (ext.gemm(x, w, *cfg, pdl).float() - ref.float()).abs().max() < 0.05
    out = {"cuBLAS": run(lambda: (pre(), F.linear(x, w))),
           "v6": run(lambda: (pre(), ext.gemm(x, w, *cfg, False))),
           "v6 + PDL": run(lambda: (pre(), ext.gemm(x, w, *cfg, True)))}
    print(f"\n{name}: GEMM kernel / predecessor start -> GEMM end (us)")
    for kname, (g_, s_) in out.items():
        print(f"  {kname:10s} {g_:6.2f}  {s_:6.2f}")
