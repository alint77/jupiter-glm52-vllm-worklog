"""Kernel time of decode_gemm variants (dgfix/old.cu: __ldg x loads,
hoisted above griddepcontrol.wait; new.cu: volatile-asm ld.global; clob.cu:
new + "memory" clobber): the GLM-5.3 verify-step linears it serves at
M = 4..32, the config decode_gemm.config picks, PDL on and off, L2 flushed
before each launch, torch-profiler median of 50, variants alternated 5 times.

    bench_dgfix.py <build root>    JSON lines on stdout
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

SHAPES = {"o_proj": (6144, 4096), "fused_qkv_a": (2624, 6144), "q_b": (4096, 2048),
          "dense gate_up": (6144, 6144), "dense down": (6144, 3072)}
HERE = Path(__file__).resolve().parent
VARIANTS = ("old", "new", "clob")
flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
dev = torch.device("cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)


def build(tag, root):
    d = Path(root) / tag
    d.mkdir(parents=True, exist_ok=True)
    return load(name=f"dg_bench_{tag}", sources=[str(HERE / "dgfix" / f"{tag}.cu")],
                extra_cuda_cflags=flags, build_directory=str(d))


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
          if e.device_type().name == "CUDA" and "decode_gemm" in e.name()]
    return st.median(ds)


def main():
    exts = {v: build(v, sys.argv[1]) for v in VARIANTS}
    from vllm.model_executor.layers.decode_gemm import config
    gen = torch.Generator(device=dev).manual_seed(0)
    for name, (n, k) in SHAPES.items():
        w = (torch.randn((n, k), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        for m in (4, 8, 16, 32):
            cfg = config(n, k, m)
            if cfg is None:
                continue
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            ref = exts["old"].gemm(x, w, *cfg, False).view(torch.int16)
            for pdl in (True, False):
                assert all(torch.equal(e.gemm(x, w, *cfg, pdl).view(torch.int16), ref)
                           for e in exts.values())
                t = {v: [] for v in VARIANTS}
                for _ in range(5):
                    for v, e in exts.items():
                        t[v].append(timed(lambda: e.gemm(x, w, *cfg, pdl)))
                row = {"shape": name, "m": m, "cfg": cfg, "pdl": pdl}
                row.update({f"{v}_us": round(st.median(t[v]), 2) for v in VARIANTS})
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
