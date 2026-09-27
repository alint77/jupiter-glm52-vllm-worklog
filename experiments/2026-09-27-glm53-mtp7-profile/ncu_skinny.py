#!/usr/bin/env python3
"""One eager o_proj-shaped call each of F.linear and skinny_gemm, for ncu.

    ncu --set full -k regex:"skinny|nvjet|gemm" --launch-skip 2 python ncu_skinny.py
"""
import sys

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.skinny_gemm import skinny_gemm

cfg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
k, n = 4096, 6144
w = (torch.randn((n, k), device="cuda") * 0.02).to(torch.bfloat16)
x = torch.randn((8, k), device="cuda").to(torch.bfloat16)
for _ in range(2):  # warm-up launches, skipped by --launch-skip
    F.linear(x, w)
    skinny_gemm(x, w, cfg)
torch.cuda.synchronize()
F.linear(x, w)
skinny_gemm(x, w, cfg)
torch.cuda.synchronize()
