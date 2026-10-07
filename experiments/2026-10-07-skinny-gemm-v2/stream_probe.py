"""Streaming ceiling for the decode GEMM weights (o_proj 48 MiB, qkv_a 31 MiB,
q_b 16 MiB): TMA bulk ring vs plain loads, vs F.linear on the same bytes.
L2 flushed (256 MB write) before every launch; kernel durations from the torch
profiler (median of 50).

    stream_probe.py
"""
import itertools
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
build = Path("/e/fscratch/profound/naeimitabiei1/caches/stream_probe")
build.mkdir(parents=True, exist_ok=True)
ext = load(name="stream_probe", sources=[str(HERE / "stream_probe.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
           build_directory=str(build))
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
    # F.linear may launch two kernels (split-K + reduce): sum per call
    per = len(ds) // reps
    return st.median(sum(ds[i * per:(i + 1) * per]) for i in range(reps))


for name, (n, k) in {"o_proj": (6144, 4096), "qkv_a": (2624, 6144), "q_b": (4096, 2048)}.items():
    w = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
    x = torch.randn(8, k, dtype=torch.bfloat16, device=dev)
    mb = w.numel() * 2 / 1e6
    res = {"F.linear": timed(lambda: F.linear(x, w))}
    for g, t, u in itertools.product((SM, 2 * SM, 4 * SM), (256, 512), (4, 8)):
        res[f"ldg g{g} t{t} u{u}"] = timed(lambda: ext.ldg(w, sink, g, t, u))
    for cpsm, chunk, stages, cons in itertools.product((1, 2), (16384, 32768, 65536), (2, 4, 6, 8),
                                                       (4,)):
        if chunk * stages * cpsm > 220 * 1024:
            continue
        res[f"tma {cpsm}/SM c{chunk >> 10}K s{stages}"] = timed(
            lambda: ext.tma(w, sink, cpsm * SM, chunk, stages, cons))
    best = sorted(res.items(), key=lambda x: x[1])
    print(f"\n{name}: {mb:.1f} MB; floor at 3.35 TB/s {mb / 3.35:.1f} us; F.linear {res['F.linear']:.1f} us")
    for key, us in best[:8]:
        print(f"  {key:24s} {us:6.2f} us  {mb / us:5.2f} TB/s")
    tma = sorted((v, k_) for k_, v in res.items() if k_.startswith("tma"))
    for us, key in tma[:5]:
        print(f"  {key:24s} {us:6.2f} us  {mb / us:5.2f} TB/s")
