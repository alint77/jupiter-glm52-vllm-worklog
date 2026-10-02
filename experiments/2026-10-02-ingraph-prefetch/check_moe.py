"""End-to-end check of tiered_prefill_moe on real routing, two tiers:
against an fp32 reference with the pipeline's own intermediate precisions
(bf16 y13, fp32 silu*up, exact w2 products, fp32 combine), and against
Marlin's fused_marlin_moe on the same weights for scale."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_routing  # noqa: E402
from vllm.model_executor.layers.fused_moe import tiered_prefill
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_quantize_weights
from vllm.scalar_type import scalar_types

torch.manual_seed(0)
dev = torch.device("cuda", 0)
H, I, DISTINCT = 6144, 2048, 6
Q = scalar_types.uint4b8


def expert(k, n):
    w = torch.randn((k, n), dtype=torch.bfloat16, device=dev) / k**0.5
    _, q, s, _, _, _ = marlin_quantize(w, Q, 32, False)
    _, codes, scales, _, _ = gptq_quantize_weights(w, Q, 32, False)
    exact = (codes.float() - 8) * scales.float().repeat_interleave(32, dim=0)
    return q, s, exact


base = [(expert(H, 2 * I), expert(I, H)) for _ in range(DISTINCT)]
smp = real_routing.samples(int(sys.argv[1]) if len(sys.argv) > 1 else 512, 1, seed=7)[0]
local_ids = np.flatnonzero(smp.local_map >= 0)       # this GPU's 64 global ids
rng = np.random.default_rng(0)
cold_ids = set(rng.choice(local_ids, 24, replace=False).tolist())
hot_map = torch.full((256,), -1, dtype=torch.int32)
cold_map = torch.full((256,), -1, dtype=torch.int32)
tiers = {"hot": [], "cold": []}
which = {}  # global id -> distinct weight index
for g in local_ids:
    name = "cold" if g in cold_ids else "hot"
    (hot_map if name == "hot" else cold_map)[g] = len(tiers[name])
    which[int(g)] = int(g) % DISTINCT
    tiers[name].append(which[int(g)])


def stack(idx):
    return {
        "w13_weight_packed": torch.stack([base[i][0][0] for i in idx]).contiguous(),
        "w13_weight_scale": torch.stack([base[i][0][1] for i in idx]).contiguous(),
        "w2_weight_packed": torch.stack([base[i][1][0] for i in idx]).contiguous(),
        "w2_weight_scale": torch.stack([base[i][1][1] for i in idx]).contiguous(),
    }


hot, cold = stack(tiers["hot"]), stack(tiers["cold"])
hot_map, cold_map = hot_map.to(dev), cold_map.to(dev)
ids = torch.from_numpy(smp.topk_ids).to(dev)
wts = torch.rand(ids.shape, device=dev).softmax(-1)
t = ids.shape[0]
x = torch.randn((t, H), dtype=torch.bfloat16, device=dev)
k_exp = tiered_prefill.prefill_scale_exponent(hot, cold)
out = tiered_prefill.tiered_prefill_moe(x, ids, wts, hot_map, cold_map, hot, cold, k_exp)
torch.accelerator.synchronize()

ref = torch.zeros((t, H), device=dev)
for g in local_ids:
    rows, ks = torch.nonzero(ids == int(g), as_tuple=True)
    if rows.numel() == 0:
        continue
    (_, _, e13), (_, _, e2) = base[which[int(g)]]
    y13 = (x[rows].float() @ e13).to(torch.bfloat16).float()
    act = torch.nn.functional.silu(y13[:, :I]) * y13[:, I:]
    ref.index_add_(0, rows, wts[rows, ks][:, None] * (act @ e2).to(torch.bfloat16).float())

# Marlin on the same weights: one call over all 64 experts
order = list(local_ids)
lmap = torch.full((256,), -1, dtype=torch.int32)
lmap[torch.tensor(order)] = torch.arange(64, dtype=torch.int32)
allw = stack([which[int(g)] for g in order])
mar = fused_marlin_moe(x, allw["w13_weight_packed"], allw["w2_weight_packed"], None, None,
                       allw["w13_weight_scale"], allw["w2_weight_scale"], wts, ids, Q.id,
                       global_num_experts=256, expert_map=lmap.to(dev),
                       workspace=marlin_make_workspace_new(dev, 4), is_k_full=True)
rms = ref.pow(2).mean().sqrt().item()
for name, got in (("ours", out.float()), ("marlin", mar.float())):
    e = (got - ref).abs()
    print(f"{name:6s}: max {e.max().item() / rms:.2e} rms-rel, mean {e.mean().item() / rms:.2e}")
e_ours = ((out.float() - ref).abs().mean() / rms).item()
e_mar = ((mar.float() - ref).abs().mean() / rms).item()
ok = e_ours <= max(e_mar, 2e-3) * 1.05 and torch.isfinite(out.float()).all()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
