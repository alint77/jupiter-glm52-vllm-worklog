"""Positive controls: skinny_gemm with one planted synchronization bug.

MUT=1 first consumer stage skips its full-barrier wait (reads before TMA lands)
MUT=2 consumers release the stage before reading it (TMA may overwrite it)
MUT=3 no named barrier between thread 0 writing `last` and the others reading
"""
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

mut = int(sys.argv[1])
here = Path(__file__).resolve().parent
build = Path("/e/fscratch/profound/naeimitabiei1/vllm-cache/torch_extensions") / f"skinny_mut{mut}"
build.mkdir(parents=True, exist_ok=True)
ext = load(name=f"skinny_mut{mut}", sources=[str(here / "skinny_mut.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a",
                              "-std=c++17", f"-DMUT={mut}"],
           extra_ldflags=["-lcuda"], build_directory=str(build))
torch.manual_seed(0)
dev = torch.device("cuda:0")
# grid 1: one CTA cycles the 8-stage ring 4 times; grid 3: split-K tiles
for m, n, k, cfg, grid in [(8, 128, 4096, 1, 1), (8, 128, 512, 1, 3)]:
    x = (torch.randn(m, k, device=dev) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device=dev) * 0.05).to(torch.bfloat16)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    ws = torch.zeros(8 * n, device=dev)
    cnt = torch.zeros((n + 63) // 64, dtype=torch.int32, device=dev)
    ext.gemm(out, x, w, ws, cnt, cfg, grid)
    want = x.float() @ w.float().t()
    err = ((out.float() - want).abs().amax() / want.abs().amax()).item()
    print(f"MUT={mut} K={k} grid={grid}: err={err:.2e}")
