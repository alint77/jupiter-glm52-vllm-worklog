"""Same-process A/B of tiered_prefill builds (define sets) at the sliced
shapes: each variant's extension is loaded side by side, one graph per variant
per routing sample, replays interleaved (ABAB...), mean wall per variant.
Build each variant once beforehand (concurrent first loads can collide).

    bench_var.py --variants "TP_WAIT1=0;;TP_XRES_BLOCKS=16" [--sizes 512,4096]
"""
import argparse
import os

import numpy as np
import torch

import bench_sliced as b
from vllm.model_executor.layers.fused_moe import tiered_prefill

ap = argparse.ArgumentParser()
ap.add_argument("--variants", required=True)
ap.add_argument("--sizes", default="512,1024,2048,4096")
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--rounds", type=int, default=5)
a = ap.parse_args()
variants = a.variants.split(";")
exts = []
for v in variants:
    os.environ["VLLM_TIERED_PREFILL_DEFINES"] = v
    tiered_prefill._extension.cache_clear()
    exts.append(tiered_prefill._extension())
orig = tiered_prefill._extension


def call(i, *args):
    tiered_prefill._extension = lambda: exts[i]
    try:
        return tiered_prefill.tiered_prefill_moe(*args)
    finally:
        tiered_prefill._extension = orig


hot, cold = b.comps(b.HOT, 0), b.comps(b.E - b.HOT, 10)
k_exp = tiered_prefill.prefill_scale_exponent(hot, cold)
perm = torch.from_numpy(np.random.default_rng(0).permutation(b.E).astype(np.int64))
hm = torch.full((b.E,), -1, dtype=torch.int32)
cm = torch.full((b.E,), -1, dtype=torch.int32)
hm[perm[:b.HOT]] = torch.arange(b.HOT, dtype=torch.int32)
cm[perm[b.HOT:]] = torch.arange(b.E - b.HOT, dtype=torch.int32)
hm, cm = hm.to(b.dev), cm.to(b.dev)
for chunk in [int(s) for s in a.sizes.split(",")]:
    tot = [[] for _ in variants]
    for smp in b.real_routing.samples(chunk, a.samples, seed=chunk):
        ids = torch.from_numpy(smp.topk_ids).to(b.dev)
        wts = torch.rand(ids.shape, device=b.dev).softmax(-1)
        x = torch.randn((chunk, b.H), dtype=torch.bfloat16, device=b.dev)
        gs = [b.graph(lambda i=i: call(i, x, ids, wts, hm, cm, hot, cold, k_exp))
              for i in range(len(variants))]
        for g in gs:
            g.replay()
        for _ in range(a.rounds):
            for i, g in enumerate(gs):
                tot[i].append(b.wall_us(g, 10))
    base = np.mean(tot[0])
    print(f"T={chunk}: " + "  ".join(
        f"[{v or 'default'}] {np.mean(t):7.1f} ({np.mean(t) - base:+6.1f})"
        for v, t in zip(variants, tot)), flush=True)
