"""Streaming rate of the tiered MoE kernel's TMA weight load (3D tensor map,
32 KiB boxes) vs plain 16 B loads, on hot-expert-sized volumes: 8 experts' w13
of one layer per rank (~151 MB at INT4 incl. nothing else) and 2 experts. L2
flushed before each launch; profiler kernel times, median of 50.

    tma_map_probe.py
"""
import itertools
import statistics as st
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
b = Path("/e/fscratch/profound/naeimitabiei1/caches/tma_map_probe")
b.mkdir(parents=True, exist_ok=True)
ext = load(name="tma_map_probe", sources=[str(HERE / "tma_map_probe.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
           extra_ldflags=["-lcuda"], build_directory=str(b))
sp = load(name="stream_probe", sources=[str(HERE.parent / "2026-10-07-skinny-gemm-v2/stream_probe.cu")],
          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
          build_directory="/e/fscratch/profound/naeimitabiei1/caches/stream_probe")
dev = torch.device("cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
sink = torch.zeros(1, dtype=torch.int32, device=dev)
SM = torch.cuda.get_device_properties(0).multi_processor_count


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
    return st.median(ds)


# INT4 w13 of one expert: 6144/16 rows of (2*2048*2*... ) -> the kernel's
# [E][HIDDEN/16][2*INTER*2] uint32 layout: 384 rows x 8192 u32 = 12.6 MB
for e in (2, 8):
    w = torch.randint(0, 1 << 30, (e, 384, 8192), dtype=torch.int32, device=dev)
    mb = w.numel() * 4 / 1e6
    res = {}
    for g, u in ((SM, 8), (2 * SM, 4)):
        res[f"16 B loads g{g} u{u}"] = timed(lambda: sp.ldg(w, sink, g, 512, u))
    for cps, stages, prod in itertools.product((1, 2), (4, 6), (1, 2, 4)):
        if cps * stages * 32 > 220:
            continue
        res[f"TMA map {cps}/SM s{stages} p{prod}"] = timed(lambda: ext.run(w, sink, cps * SM, stages, prod))
    print(f"\n{e} experts' w13, {mb:.1f} MB")
    for k_, us in sorted(res.items(), key=lambda x: x[1]):
        print(f"  {k_:26s} {us:7.2f} us  {mb / us:5.2f} TB/s")
