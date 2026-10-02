"""Milestone 1 correctness: wgmma W4A16 GEMM on Marlin-layout weights vs
X @ w_ref (marlin_quantize's dequantized weights) in fp32, rounded to bf16."""

import sys

import torch

from vllm.model_executor.layers.fused_moe import tiered_prefill
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types

torch.manual_seed(0)
dev = torch.device("cuda", 0)
ok = True
for k, f in ((6144, 4096), (2048, 6144)):
    refs, qs, ss = [], [], []
    for _ in range(2):
        w = torch.randn((k, f), dtype=torch.bfloat16, device=dev) / k**0.5
        w_ref, q, s, _, _, _ = marlin_quantize(w, scalar_types.uint4b8, 32, False)
        refs.append(w_ref)
        qs.append(q)
        ss.append(s)
    q = torch.stack(qs).contiguous()
    s = torch.stack(ss).contiguous()
    for n in (1, 7, 16, 24, 33, 64):
        x = torch.randn((n, k), dtype=torch.bfloat16, device=dev)
        y = tiered_prefill.dense(x, q, s)
        torch.accelerator.synchronize()
        for e in range(2):
            ref = (x.float() @ refs[e].float()).to(torch.bfloat16).float()
            got = y[e].float()
            err = (got - ref).abs()
            rel = (err / ref.abs().clamp_min(1e-3)).max().item()
            bad = (err > 0.02 * ref.abs() + 1e-2).sum().item()
            print(f"K={k} F={f} N={n} e={e}: max|d| {err.max().item():.4g} "
                  f"(|ref| max {ref.abs().max().item():.3g}), max rel {rel:.3g}, "
                  f"{bad} outside tolerance")
            ok &= bad == 0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
