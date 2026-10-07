"""HBM read: kernel time vs bytes for plain 16 B loads (132 x 512 threads,
8 in flight) and the MoE's TMA tensor-map load (1 CTA/SM, 4 stages), 10 MB to
1 GB, L2 flushed before each launch; least-squares fit t = fixed + bytes / BW.

    bw_sweep.py
"""
import statistics as st
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
tma = load(name="tma_map_probe", sources=[str(HERE / "tma_map_probe.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
           extra_ldflags=["-lcuda"], build_directory="/e/fscratch/profound/naeimitabiei1/caches/tma_map_probe")
sp = load(name="stream_probe", sources=[str(HERE.parent / "2026-10-07-skinny-gemm-v2/stream_probe.cu")],
          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
          build_directory="/e/fscratch/profound/naeimitabiei1/caches/stream_probe")
dev = torch.device("cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
sink = torch.zeros(1, dtype=torch.int32, device=dev)
SM = torch.cuda.get_device_properties(0).multi_processor_count


def timed(fn, reps=20):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            flush.fill_(1)
            fn()
        torch.cuda.synchronize()
    return st.median(e.duration_ns() / 1e3 for e in p.profiler.kineto_results.events()
                     if e.device_type().name == "CUDA" and "Fill" not in e.name())


rows = {"plain 16 B loads": [], "TMA tensor map": []}
for experts in (1, 2, 4, 8, 16, 32, 80):  # x 12.6 MB (one expert's INT4 w13)
    w = torch.randint(0, 1 << 30, (experts, 384, 8192), dtype=torch.int32, device=dev)
    mb = w.numel() * 4 / 1e6
    rows["plain 16 B loads"].append((mb, timed(lambda: sp.ldg(w, sink, SM, 512, 8))))
    rows["TMA tensor map"].append((mb, timed(lambda: tma.run(w, sink, SM, 4, 1))))
    del w
for name, pts in rows.items():
    mb = np.array([p[0] for p in pts])
    us = np.array([p[1] for p in pts])
    slope, fixed = np.polyfit(mb, us, 1)
    print(f"{name}: t = {fixed:.2f} us + MB / {1 / slope:.2f} TB/s   "
          + "  ".join(f"{m:.0f}MB:{t:.1f}us({m / t:.2f})" for m, t in pts))
