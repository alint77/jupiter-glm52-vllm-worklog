"""Milestone 1 correctness: wgmma W4A16 GEMM on Marlin-layout weights vs
X @ ((code - 8) * scale) in fp32 (exact products, as the kernel): no output
more than one bf16 ulp off, and under 1e-4 of them beyond half an ulp (the
tensor cores' fp32 accumulation can tip a value at a rounding midpoint). The exact weights are checked to round to marlin_quantize's w_ref."""

import sys

import torch

from vllm.model_executor.layers.fused_moe import tiered_prefill
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_quantize_weights,
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
        _, codes, scales, _, _ = gptq_quantize_weights(
            w, scalar_types.uint4b8, 32, False
        )
        exact = (codes.float() - 8) * scales.float().repeat_interleave(32, dim=0)
        assert torch.equal(exact.to(torch.bfloat16), w_ref), "exact != w_ref"
        refs.append(exact)
        qs.append(q)
        ss.append(s)
    q = torch.stack(qs).contiguous()
    s = torch.stack(ss).contiguous()
    for n in (1, 7, 16, 24, 33, 64, 100, 128, 200):
        x = torch.randn((n, k), dtype=torch.bfloat16, device=dev)
        y = tiered_prefill.dense(x, q, s)
        torch.accelerator.synchronize()
        for e in range(2):
            ref = x.float() @ refs[e].float()
            got = y[e].float()
            err = (got - ref).abs()
            rel = (err / ref.abs().clamp_min(1e-3)).max().item()
            half = (err > 2**-8 * ref.abs() * 1.01 + 1e-5).float().mean().item()
            bad = (err > 2**-7 * ref.abs() * 1.01 + 1e-5).sum().item()
            bad += int(half > 1e-4)
            print(f"K={k} F={f} N={n} e={e}: max|d| {err.max().item():.4g} "
                  f"(|ref| max {ref.abs().max().item():.3g}), max rel {rel:.3g}, "
                  f"{bad} outside tolerance")
            ok &= bad == 0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
