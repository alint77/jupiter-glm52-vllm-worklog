"""Replay a dumped serving call (nan-r<rank>.pt) through tiered_prefill_moe and
fused Marlin; localise any non-finite output (rows, experts, stage)."""
import sys

import torch

from vllm.model_executor.layers.fused_moe import tiered_prefill

d = torch.load(sys.argv[1], weights_only=False)
dev = torch.device("cuda", 0)
g = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()
     if k not in ("hot", "cold")}
hot = {k: v.to(dev) for k, v in d["hot"].items()}
cold = {k: v.to(dev) for k, v in d["cold"].items()}
x, ids, wts = g["x"], g["ids"], g["wts"]
print(d["layer"], "tokens", x.shape[0], "staged", d["staged"], "exp", d["exp"],
      "hot", hot["w13_weight_packed"].shape[0], "cold", cold["w13_weight_packed"].shape[0])
print("dumped out nonfinite rows", (~d["out"].isfinite()).any(1).sum().item(),
      "marlin nonfinite rows", (~d["marlin"].isfinite()).any(1).sum().item())
for name, t in list(hot.items()) + list(cold.items()):
    if t.dtype.is_floating_point:
        print(name, t.shape, "nonfinite", (~t.isfinite()).sum().item(),
              "absmax", t.float().abs().max().item(), "min>0", t.float().abs()[t != 0].min().item())
print("x absmax", x.float().abs().max().item(), "row absmax min", x.float().abs().amax(1).min().item())
print("recomputed exp", tiered_prefill.prefill_scale_exponent(hot, cold))
out = tiered_prefill.tiered_prefill_moe(x, ids, wts, g["hmap"], g["cmap"], hot, cold, d["exp"])
torch.accelerator.synchronize()
bad = (~out.isfinite()).any(1)
print("replay nonfinite rows", bad.sum().item())
if bad.any():
    r = bad.nonzero()[:5, 0]
    print("bad rows", r.tolist(), "ids", ids[r].tolist())
    loc = torch.where(g["hmap"][ids.long().clamp_min(0)] >= 0, 0, torch.where(g["cmap"][ids.long().clamp_min(0)] >= 0, 1, -1))
    print("tier of routes", loc[r].tolist())
fin = out.isfinite().all(1) & d["marlin"].to(dev).isfinite().all(1)
ref = d["marlin"].to(dev)[fin].float()
print("vs marlin on finite rows: mean err / rms",
      ((out[fin].float() - ref).abs().mean() / ref.pow(2).mean().sqrt()).item())
