"""MoE outputs at the sliced shapes for a fixed set of inputs, saved for a
bitwise comparison between kernel versions:
    bitwise.py save <file>      (run under each tree)
    bitwise.py cmp <file a> <file b>
Sizes 1..4096 with real routing (random below 64 tokens), plus a call where a
quarter of the routes are -1 and one where every route goes to one expert."""
import sys

import numpy as np
import torch

if sys.argv[1] == "cmp":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    bad = 0
    for k in a:
        same = torch.equal(a[k], b[k])
        bad += not same
        if not same:
            d = (a[k].float() - b[k].float()).abs()
            print(f"{k}: DIFFER max {d.max().item():.3g} frac {(d > 0).float().mean().item():.3g}")
    print(f"{len(a) - bad}/{len(a)} identical")
    sys.exit(bad != 0)

import bench_sliced as b  # noqa: E402
from vllm.model_executor.layers.fused_moe import tiered_prefill  # noqa: E402

hot, cold = b.comps(b.HOT, 0), b.comps(b.E - b.HOT, 10)
k_exp = tiered_prefill.prefill_scale_exponent(hot, cold)
perm = torch.from_numpy(np.random.default_rng(0).permutation(b.E).astype(np.int64))
hm = torch.full((b.E,), -1, dtype=torch.int32)
cm = torch.full((b.E,), -1, dtype=torch.int32)
hm[perm[:b.HOT]] = torch.arange(b.HOT, dtype=torch.int32)
cm[perm[b.HOT:]] = torch.arange(b.E - b.HOT, dtype=torch.int32)
hm, cm = hm.to(b.dev), cm.to(b.dev)
out = {}
g = torch.Generator(device=b.dev).manual_seed(1)
for chunk in (1, 7, 33, 64, 200, 512, 1000, 2048, 3000, 4096):
    if chunk >= 64:
        ids = torch.from_numpy(next(iter(b.real_routing.samples(chunk, 1, seed=chunk))).topk_ids).to(b.dev)
    else:
        ids = torch.stack([torch.randperm(b.E, device=b.dev, generator=g)[:8] for _ in range(chunk)])
    wts = torch.rand(ids.shape, device=b.dev, generator=g).softmax(-1)
    x = torch.randn((chunk, b.H), dtype=torch.bfloat16, device=b.dev, generator=g)
    x[0] *= 1e-20 if chunk > 1 else 1.0
    out[f"real{chunk}"] = tiered_prefill.tiered_prefill_moe(x, ids, wts, hm, cm, hot, cold, k_exp, 0).cpu()
    if chunk in (200, 4096):
        ids2 = ids.clone()
        ids2[torch.rand(ids.shape, device=b.dev, generator=g) < 0.25] = -1
        out[f"holes{chunk}"] = tiered_prefill.tiered_prefill_moe(x, ids2, wts, hm, cm, hot, cold, k_exp, 0).cpu()
        ids3 = torch.full_like(ids, int(perm[3]))
        ids3[:, 1:] = -1
        out[f"one{chunk}"] = tiered_prefill.tiered_prefill_moe(x, ids3, wts, hm, cm, hot, cold, k_exp, 0).cpu()
torch.save(out, sys.argv[2])
print("saved", len(out))
