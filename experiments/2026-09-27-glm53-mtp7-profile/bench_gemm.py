#!/usr/bin/env python3
"""Per-projection GEMM time at the verify step's M=8, against its HBM floor.

GLM-5.3 at TP4, bf16 weights as served. Each case is a CUDA graph of 64 calls
cycling over enough weight copies to defeat L2; time per call from graph
replay. `floor` is weight bytes / the measured streaming read bandwidth.

    bench_gemm.py [--m 8] [--copies 8]
"""

import argparse
import json

import torch

# name: (K, N, calls per step, dtype) -- per-rank shapes
CASES = {
    "fused_qkv_a (replicated)": (6144, 2048 + 576, 78, torch.bfloat16),
    "q_b": (2048, 16384 // 4, 78, torch.bfloat16),
    "o_proj": (16384 // 4, 6144, 78, torch.bfloat16),
    "shared gate_up": (6144, 2 * 2048 // 4, 75, torch.bfloat16),
    "shared down": (2048 // 4, 6144, 75, torch.bfloat16),
    "router (fp32)": (6144, 256, 75, torch.float32),
    "dense gate_up": (6144, 2 * 12288 // 4, 3, torch.bfloat16),
    "dense down": (12288 // 4, 6144, 3, torch.bfloat16),
    "indexer wq_b (replicated)": (2048, 32 * 128, 21, torch.bfloat16),
    "indexer wk+weights (replicated)": (6144, 128 + 32, 21, torch.bfloat16),
    "lm_head": (6144, 154880 // 4, 1, torch.bfloat16),
}


def graph_time(fn, reps: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1000  # us per graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--calls", type=int, default=64)
    ap.add_argument("--ll", action="store_true", help="also time cute-DSL ll_bf16")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    # streaming read bandwidth: sum over a 4 GiB buffer
    big = torch.empty(1 << 31, dtype=torch.bfloat16, device=dev)
    out = torch.empty((), dtype=torch.float32, device=dev)
    t = graph_time(lambda: torch.sum(big, dim=0, dtype=torch.float32, out=out))
    bw = big.numel() * 2 / (t * 1e-6)
    del big
    print(f"streaming read: {bw / 1e12:.2f} TB/s")
    rows, total_now, total_floor = [], 0.0, 0.0
    for name, (k, n, per_step, dtype) in CASES.items():
        ws = [torch.randn((n, k), device=dev, dtype=dtype) * 0.02 for _ in range(args.copies)]
        x = torch.randn((args.m, k), device=dev, dtype=dtype)

        def run():
            for i in range(args.calls):
                torch.nn.functional.linear(x, ws[i % args.copies])

        us = graph_time(run) / args.calls
        ll = float("nan")
        if dtype == torch.bfloat16 and args.ll:
            from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

            def run_ll():
                for i in range(args.calls):
                    ll_bf16_gemm(x, ws[i % args.copies], torch.float32)

            try:
                ll = graph_time(run_ll) / args.calls
            except Exception as error:  # a shape the kernel does not take
                print(f"  ll_bf16 {name}: {error!r:.120}")
        floor = n * k * ws[0].element_size() / bw * 1e6
        total_now += us * per_step
        total_floor += floor * per_step
        rows.append({"name": name, "k": k, "n": n, "us": us, "ll_us": ll, "floor_us": floor,
                     "per_step": per_step, "step_ms": us * per_step / 1000,
                     "floor_step_ms": floor * per_step / 1000})
        print(f"{name:32s} K={k:5d} N={n:6d}  {us:7.1f} us  floor {floor:6.1f}  "
              f"({floor / us:4.0%})  x{per_step:3d} = {us * per_step / 1000:5.2f} ms "
              f"(floor {floor * per_step / 1000:5.2f})  ll_bf16 {ll:6.1f} us", flush=True)
        del ws
    print(f"total {total_now / 1000:.2f} ms/step, floor {total_floor / 1000:.2f}")
    print(json.dumps({"bw": bw, "rows": rows}))


if __name__ == "__main__":
    main()
