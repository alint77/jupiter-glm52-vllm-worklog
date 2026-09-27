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
        m = 8
        x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
        ref = x.float() @ ws[0].float().t()
        scale = ref.abs().max().item()
        calls = 64
        t_l = graph_us(lambda: [F.linear(x, ws[i % 8]) for i in range(calls)], calls)
        cells = []
        best = None
        for cfg in (0, 1, 2, 3):
            for grid in ((0, 66, 264) if name == "o_proj" else (0,)):
                err = (skinny_gemm(x, ws[0], cfg, grid).float() - ref).abs().max().item()
                t_s = graph_us(lambda: [skinny_gemm(x, ws[i % 8], cfg, grid)
                                        for i in range(calls)], calls)
                cells.append(f"c{cfg}g{grid or 132}:{t_s:5.1f}({err / scale:.0e})")
                best = t_s if best is None else min(best, t_s)
        save_now += (t_l - best) * per_step / 1000
        save_floor += (t_l - floor) * per_step / 1000
        print(f"{name:15s} K={k:5d} N={n:5d} floor {floor:5.1f}  F.linear {t_l:5.1f}  "
              f"best skinny {best:5.1f}  | " + " ".join(cells), flush=True)
    print(f"M=8, per verify step at these call counts: skinny saves {save_now:.2f} ms "
          f"over F.linear (floor would save {save_floor:.2f})")


if __name__ == "__main__":
    main()
