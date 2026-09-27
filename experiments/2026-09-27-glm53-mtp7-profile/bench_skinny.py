#!/usr/bin/env python3
"""skinny_gemm against F.linear at the verify step's shapes, M = 1 and 8.

Correctness: max error against an fp32 reference, relative to the output's
max, next to F.linear's own error. Speed: CUDA graphs of 64 calls cycling
over 8 weight copies (no L2 reuse), graph replay time per call, against the
HBM floor from a streaming read.

    bench_skinny.py
"""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.skinny_gemm import skinny_gemm

SHAPES = {  # name: (K, N, calls per step)
    "fused_qkv_a": (6144, 2624, 78),
    "q_b": (2048, 4096, 78),
    "o_proj": (4096, 6144, 78),
    "shared gate_up": (6144, 1024, 75),
    "shared down": (512, 6144, 75),
    "dense gate_up": (6144, 6144, 3),
    "dense down": (3072, 6144, 3),
    "indexer wq_b": (2048, 4096, 21),
    "indexer wk": (6144, 160, 21),
}


def graph_us(fn, calls: int) -> float:
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
    for _ in range(20):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 20 * 1000 / calls


def main() -> None:
    dev = torch.device("cuda:0")
    big = torch.empty(1 << 31, dtype=torch.bfloat16, device=dev)
    out = torch.empty((), dtype=torch.float32, device=dev)
    bw = big.numel() * 2 / (graph_us(lambda: torch.sum(big, dim=0, dtype=torch.float32,
                                                     out=out), 1) * 1e-6)
    del big
    print(f"streaming read {bw / 1e12:.2f} TB/s")
    gen = torch.Generator(device=dev).manual_seed(0)
    save_now = save_floor = 0.0
    for name, (k, n, per_step) in SHAPES.items():
        ws = [(torch.randn((n, k), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
              for _ in range(8)]
        floor = n * k * 2 / bw * 1e6
        for m in (1, 8):
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            ref = x.float() @ ws[0].float().t()
            scale = ref.abs().max().item()
            err = ((skinny_gemm(x, ws[0]).float() - ref).abs().max().item()) / scale
            err_lib = ((F.linear(x, ws[0]).float() - ref).abs().max().item()) / scale
            again = skinny_gemm(x, ws[0])
            stable = torch.equal(again, skinny_gemm(x, ws[0]))
            calls = 64
            t_s = graph_us(lambda: [skinny_gemm(x, ws[i % 8]) for i in range(calls)], calls)
            t_l = graph_us(lambda: [F.linear(x, ws[i % 8]) for i in range(calls)], calls)
            if m == 8:
                save_now += (t_l - t_s) * per_step / 1000
                save_floor += (t_l - floor) * per_step / 1000
            print(f"{name:15s} M={m} K={k:5d} N={n:5d}  skinny {t_s:6.1f} us  "
                  f"F.linear {t_l:6.1f}  floor {floor:6.1f} ({floor / t_s:4.0%})  "
                  f"err {err:.1e} (lib {err_lib:.1e}) repeat-equal {stable}", flush=True)
    print(f"M=8, per verify step at these call counts: skinny saves {save_now:.2f} ms "
          f"over F.linear (floor would save {save_floor:.2f})")


if __name__ == "__main__":
    main()
