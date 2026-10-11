"""One tiered_prefill_moe call at the sliced shapes per size, inside
cudaProfilerStart/Stop (for ncu --profile-from-start off); setup as
bench_sliced.py.  one_call.py 512,4096"""
import sys

import numpy as np
import torch

import bench_sliced as b
from vllm.model_executor.layers.fused_moe import tiered_prefill

sizes = [int(s) for s in sys.argv[1].split(",")]
hot, cold = b.comps(b.HOT, 0), b.comps(b.E - b.HOT, 10)
k_exp = tiered_prefill.prefill_scale_exponent(hot, cold)
perm = torch.from_numpy(np.random.default_rng(0).permutation(b.E).astype(np.int64))
hm = torch.full((b.E,), -1, dtype=torch.int32)
cm = torch.full((b.E,), -1, dtype=torch.int32)
hm[perm[:b.HOT]] = torch.arange(b.HOT, dtype=torch.int32)
cm[perm[b.HOT:]] = torch.arange(b.E - b.HOT, dtype=torch.int32)
hm, cm = hm.to(b.dev), cm.to(b.dev)
for chunk in sizes:
    smp = next(iter(b.real_routing.samples(chunk, 1, seed=chunk)))
    ids = torch.from_numpy(smp.topk_ids).to(b.dev)
    wts = torch.rand(ids.shape, device=b.dev).softmax(-1)
    x = torch.randn((chunk, b.H), dtype=torch.bfloat16, device=b.dev)
    f = lambda: tiered_prefill.tiered_prefill_moe(x, ids, wts, hm, cm, hot, cold, k_exp, 0)
    f()
    torch.accelerator.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    f()
    torch.accelerator.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
