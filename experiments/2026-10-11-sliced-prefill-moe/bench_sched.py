"""Same-process A/B of tiered_prefill_moe schedules at the sliced shapes:
per size and routing sample, one graph per schedule, replays interleaved
(ABAB...), mean wall per schedule.   bench_sched.py --schedules 0,1"""
import argparse

import numpy as np
import torch

import bench_sliced as b
from vllm.model_executor.layers.fused_moe import tiered_prefill

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="512,1024,2048,4096")
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--schedules", default="0,1")
ap.add_argument("--rounds", type=int, default=5)
a = ap.parse_args()
scheds = [int(s) for s in a.schedules.split(",")]
hot, cold = b.comps(b.HOT, 0), b.comps(b.E - b.HOT, 10)
k_exp = tiered_prefill.prefill_scale_exponent(hot, cold)
perm = torch.from_numpy(np.random.default_rng(0).permutation(b.E).astype(np.int64))
hm = torch.full((b.E,), -1, dtype=torch.int32)
cm = torch.full((b.E,), -1, dtype=torch.int32)
hm[perm[:b.HOT]] = torch.arange(b.HOT, dtype=torch.int32)
cm[perm[b.HOT:]] = torch.arange(b.E - b.HOT, dtype=torch.int32)
hm, cm = hm.to(b.dev), cm.to(b.dev)
for chunk in [int(s) for s in a.sizes.split(",")]:
    tot = {s: [] for s in scheds}
    for smp in b.real_routing.samples(chunk, a.samples, seed=chunk):
        ids = torch.from_numpy(smp.topk_ids).to(b.dev)
        wts = torch.rand(ids.shape, device=b.dev).softmax(-1)
        x = torch.randn((chunk, b.H), dtype=torch.bfloat16, device=b.dev)
        gs = {s: b.graph(lambda s=s: tiered_prefill.tiered_prefill_moe(
            x, ids, wts, hm, cm, hot, cold, k_exp, s)) for s in scheds}
        for _ in range(a.rounds):
            for s in scheds:
                tot[s].append(b.wall_us(gs[s], 10))
    base = np.mean(tot[scheds[0]])
    print(f"T={chunk}: " + "  ".join(
        f"sched {s} {np.mean(v):7.1f} us ({np.mean(v) - base:+6.1f})" for s, v in tot.items()), flush=True)
