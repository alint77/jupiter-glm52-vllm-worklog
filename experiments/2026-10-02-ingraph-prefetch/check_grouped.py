"""Grouped GEMM correctness on real routing: each sorted route row against
x[token] @ ((code - 8) * scale) of its expert, exact fp32 reference."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_routing  # noqa: E402
from vllm.model_executor.layers.fused_moe import tiered_prefill
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_quantize_weights
from vllm.scalar_type import scalar_types

torch.manual_seed(0)
dev = torch.device("cuda", 0)
E = 64
ok = True
for k, f in ((6144, 4096), (2048, 6144)):
    # 4 distinct experts, tiled over 64 slots (indexing by expert is checked)
    exact, qs, ss = [], [], []
    for _ in range(4):
        w = torch.randn((k, f), dtype=torch.bfloat16, device=dev) / k**0.5
        _, q, s, _, _, _ = marlin_quantize(w, scalar_types.uint4b8, 32, False)
        _, codes, scales, _, _ = gptq_quantize_weights(w, scalar_types.uint4b8, 32, False)
        exact.append((codes.float() - 8) * scales.float().repeat_interleave(32, dim=0))
        qs.append(q)
        ss.append(s)
    q = torch.stack([qs[e % 4] for e in range(E)]).contiguous()
    s = torch.stack([ss[e % 4] for e in range(E)]).contiguous()
    for chunk in (512, 2048):
        smp = real_routing.samples(chunk, 1, seed=chunk)[0]
        ids = torch.from_numpy(smp.topk_ids).to(dev)
        lmap = torch.from_numpy(smp.local_map).to(dev)
        x = torch.randn((chunk, k), dtype=torch.bfloat16, device=dev)
        y, token, expert = tiered_prefill.grouped_gemm(x, ids, lmap, q, s)
        torch.accelerator.synchronize()
        worst = 0.0
        bad = 0
        for e4 in range(4):
            sel = (expert % 4) == e4
            ref = x[token[sel]].float() @ exact[e4]
            err = (y[sel].float() - ref).abs()
            bad += int((err > 2**-7 * ref.abs() * 1.01 + 1e-5).sum().item())
            worst = max(worst, (err / ref.abs().clamp_min(1e-2)).max().item())
        print(f"K={k} chunk={chunk}: {y.shape[0]} routes, max rel {worst:.3g}, {bad} beyond 1 ulp")
        ok &= bad == 0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
