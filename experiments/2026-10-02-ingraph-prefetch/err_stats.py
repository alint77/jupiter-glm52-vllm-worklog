"""Error vs the exact fp32 reference ((code-8)*scale): this kernel against
Marlin's numerics (bf16-rounded weights, i.e. x @ w_ref in fp32)."""
import torch
from vllm.model_executor.layers.fused_moe import tiered_prefill
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_quantize_weights
from vllm.scalar_type import scalar_types
torch.manual_seed(0)
dev = torch.device("cuda", 0)
for k, f in ((6144, 4096), (2048, 6144)):
    w = torch.randn((k, f), dtype=torch.bfloat16, device=dev) / k**0.5
    w_ref, q, s, _, _, _ = marlin_quantize(w, scalar_types.uint4b8, 32, False)
    _, codes, scales, _, _ = gptq_quantize_weights(w, scalar_types.uint4b8, 32, False)
    exact = (codes.float() - 8) * scales.float().repeat_interleave(32, dim=0)
    for n in (16, 128):
        x = torch.randn((n, k), dtype=torch.bfloat16, device=dev)
        ref = x.float() @ exact
        rms = ref.pow(2).mean().sqrt().item()
        ours = tiered_prefill.dense(x, q[None].contiguous(), s[None].contiguous())[0].float()
        marlin = (x.float() @ w_ref.float()).to(torch.bfloat16).float()
        for name, got in (("ours", ours), ("marlin numerics", marlin)):
            e = (got - ref).abs()
            print(f"K={k} N={n} {name:16s}: max {e.max().item() / rms:.2e} rms-rel, "
                  f"mean {e.mean().item() / rms:.2e}, >1ulp-equivalent "
                  f"{(e > 2**-8 * ref.abs() * 1.01 + 1e-5).float().mean().item():.1e}")
