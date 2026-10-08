"""decode_gemm (this tree) vs F.linear for the GLM-5.3 verify-step bf16
linears (TP4 shards) at M = 1..32: every (warps, unroll) config the extension
compiles, L2 flushed before each launch, torch-profiler median of 30, max
error against an fp32 reference (relative to the output's max).

    bench_gemm_m32.py [--m 1 8 16 24 32]     JSON lines on stdout
"""
import argparse
import json
import statistics as st

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from vllm.model_executor.layers.decode_gemm import _extension

SHAPES = {  # name: (N, K)
    "o_proj": (6144, 4096), "fused_qkv_a": (2624, 6144), "q_b": (4096, 2048),
    "dense gate_up": (6144, 6144), "dense down": (6144, 3072),
}
CONFIGS = [(2, 4, 1), (2, 8, 1), (4, 2, 1), (4, 4, 1), (8, 2, 1), (8, 4, 1), (16, 2, 1),
           (2, 2, 2), (2, 4, 2), (4, 2, 2), (4, 4, 2), (8, 2, 2), (2, 2, 4), (4, 2, 4)]
dev = torch.device("cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)


def timed(fn, reps=30):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[1, 8, 9, 16, 24, 32])
    a = ap.parse_args()
    ext = _extension()
    gen = torch.Generator(device=dev).manual_seed(0)
    for name, (n, k) in SHAPES.items():
        w = (torch.randn((n, k), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        for m in a.m:
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            ref = x.float() @ w.float().t()
            scale = ref.abs().max().item()
            row = {"shape": name, "n": n, "k": k, "m": m,
                   "cublas": round(timed(lambda: F.linear(x, w)), 2)}
            best = None
            for wp, u, r in CONFIGS:
                if k % (32 * wp * u) or n % (16 * r):
                    continue
                y = ext.gemm(x, w, wp, u, r, False)
                err = (y.float() - ref).abs().max().item() / scale
                us = timed(lambda: ext.gemm(x, w, wp, u, r, False))
                key = f"{wp}x{u}r{r}"
                row[key] = round(us, 2)
                if err > 1e-2:
                    row[key + "_err"] = err
                elif best is None or us < best[0]:
                    best = (us, key)
            row["best"] = best[1] if best else None
            row["best_us"] = round(best[0], 2) if best else None
            row["floor_us"] = round(n * k * 2 / 3.64e6, 2)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
