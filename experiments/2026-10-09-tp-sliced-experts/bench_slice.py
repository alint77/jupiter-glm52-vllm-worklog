#!/usr/bin/env python3
"""Tiered decode MoE on whole experts vs 4-way intermediate slices.

Builds `tiered_decode.cu` here (TD_INTER / TD_CHUNK_K / TD_STAGES; production
defaults reproduce the shipped kernel) as separate extensions.

  --check   whole experts (2048) against the sum of the four 512-wide slices
            of the same checkpoint experts, and both against fp32, at M=8/32
  --time    one variant over (hot, cold) cells at M tokens: whole experts at
            per-GPU counts (EP: 8 * M routes, the rest on other GPUs), slices at
            node-wide counts (all 8 * M routes local). Graph replay of 20
            calls with distinct routings, best of 10, us per call.

    bench_slice.py --check
    bench_slice.py --time --variant slice-512x8 --m 8 --cells 40,4 ...

Run NUMA-bound on the GPU's Grace node.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HIDDEN, TOPK = 6144, 8
VARIANTS = {  # name: build defines (production defaults otherwise)
    "full-1024x4": "TD_INTER=2048",
    "full-512x8": "TD_INTER=2048 TD_CHUNK_K=512 TD_STAGES=8",
    "full-1024x4-noflush": "TD_INTER=2048 TD_NO_FLUSH",
    "slice-512x8": "TD_INTER=512 TD_CHUNK_K=512 TD_STAGES=8",
    "slice-1024x4-512x8": "TD_INTER=512 TD_CHUNK0=1024 TD_STAGES0=4 TD_CHUNK1=512 TD_STAGES1=8",
    "slice-1024x4-512x6": "TD_INTER=512 TD_CHUNK0=1024 TD_STAGES0=4 TD_CHUNK1=512 TD_STAGES1=6",
    "slice-1024x4-512x8-noflush":
        "TD_INTER=512 TD_CHUNK0=1024 TD_STAGES0=4 TD_CHUNK1=512 TD_STAGES1=8 TD_NO_FLUSH",
}


def inter_of(variant):
    return int(VARIANTS[variant].split("TD_INTER=")[1].split()[0])


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("tiered_test", REPO / "tests/kernels/moe/test_tiered_decode_moe.py")


def extension(variant):
    import vllm.model_executor.layers.fused_moe.tiered_decode as td

    os.environ["VLLM_TIERED_DECODE_DEFINES"] = "TD_SLICEBENCH " + VARIANTS[variant]
    td._HERE = HERE
    td._extension.cache_clear()
    mod = td._extension()
    ws = torch.zeros(mod.workspace_bytes(), dtype=torch.uint8, device="cuda")
    return mod, ws


def parts(tier):
    if tier is None:
        e = torch.empty(0, device="cuda")
        return e, e, e, e
    return (tier["w13_weight_packed"], tier["w13_weight_scale"],
            tier["w2_weight_packed"], tier["w2_weight_scale"])


def forward(ext, x, ids, w, hot_map, cold_map, hot, cold):
    mod, ws = ext
    e = torch.empty(0, device=x.device)
    out = torch.empty_like(x)
    mod.forward(out, x, ids, w, hot_map, cold_map, e, e, e, 0, False, *parts(hot),
                *parts(cold), ws, True, e)
    return out


def slice_ckpt(codes13, codes2, s13, s2, r, inter=2048, ways=4):
    w = inter // ways
    rows = torch.cat([torch.arange(r * w, (r + 1) * w),
                      inter + torch.arange(r * w, (r + 1) * w)])
    return (codes13[:, rows].contiguous(), codes2[:, :, r * w:(r + 1) * w].contiguous(),
            s13[:, rows].contiguous(), s2[:, :, r * w // 32:(r + 1) * w // 32].contiguous())


def check():
    dev = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(0)
    n_hot, n_cold = 6, 2
    ck = T._int4_experts(n_hot + n_cold, gen)
    full = T._int4_marlin_tier(*ck, dev)
    slices = [T._int4_marlin_tier(*slice_ckpt(*ck, r), dev) for r in range(4)]

    def split(tier):
        hot = {k: v[:n_hot].contiguous() for k, v in tier.items()}
        cold = T._to_grace({k: v[n_hot:].contiguous() for k, v in tier.items()}, dev)
        return hot, cold

    full_t = split(full)
    slice_t = [split(s) for s in slices]
    ext_full = extension("full-1024x4")
    ext_slice = {v: extension(v) for v in VARIANTS
                 if v.startswith("slice") and "noflush" not in v}
    w13 = T._int4_dequant(ck[0], ck[2])  # [E, 2I, H]
    w2 = T._int4_dequant(ck[1], ck[3])   # [E, H, I]
    for m in (8, 32):
        g = torch.Generator().manual_seed(m)
        x = (torch.randn((m, HIDDEN), generator=g) * 0.3).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(n_hot + n_cold, generator=g)[:TOPK]
                           for _ in range(m)]).to(torch.int32)
        wt = torch.rand((m, TOPK), generator=g)
        hot_map = torch.full((16,), -1, dtype=torch.int32)
        cold_map = torch.full((16,), -1, dtype=torch.int32)
        hot_map[:n_hot] = torch.arange(n_hot, dtype=torch.int32)
        cold_map[n_hot:n_hot + n_cold] = torch.arange(n_cold, dtype=torch.int32)
        args = [t.to(dev) for t in (x, ids, wt, hot_map, cold_map)]
        ref = torch.zeros((m, HIDDEN))
        xf = x.float()
        for t in range(m):
            for k in range(TOPK):
                e = int(ids[t, k])
                h = w13[e] @ xf[t]
                a = torch.nn.functional.silu(h[:2048]) * h[2048:]
                ref[t] += wt[t, k] * (w2[e] @ a)
        got_full = forward(ext_full, *args, *full_t).float().cpu()
        res = {"m": m, "full_vs_fp32": float((got_full - ref).abs().max() / ref.abs().max())}
        for v, ext in ext_slice.items():
            s = sum(forward(ext, *args, *st).float() for st in slice_t).cpu()
            res[f"{v}_vs_fp32"] = float((s - ref).abs().max() / ref.abs().max())
            res[f"{v}_vs_full"] = float((s - got_full).abs().max() / ref.abs().max())
        print(json.dumps(res), flush=True)


def ep_routing(m, h, c, gen, ntok_p, glob=512):
    """EP: h hot (ids 0..) and c cold (ids 256..) experts on this GPU, the
    rest of the 8 * m routes to ids owned elsewhere (not in either map)."""
    ids = torch.full((m, TOPK), -1, dtype=torch.int32)
    fill = torch.zeros(m, dtype=torch.int64)
    for e in list(range(h)) + list(range(256, 256 + c)):
        n = 1 + int(torch.multinomial(ntok_p, 1, generator=gen))
        for t in torch.randperm(m, generator=gen).tolist():
            if n == 0:
                break
            if fill[t] < TOPK:
                ids[t, fill[t]] = e
                fill[t] += 1
                n -= 1
    remote = 128
    for t in range(m):
        while fill[t] < TOPK:
            ids[t, fill[t]] = remote
            remote = remote + 1 if remote < 255 else 128
            fill[t] += 1
    return ids


def all_routing(m, h, c, gen):
    """Slices: every one of the 8 * m routes lands on one of the h + c touched
    experts (each touched at least once, no token twice on one expert)."""
    experts = list(range(h)) + list(range(256, 256 + c))
    ids = torch.full((m, TOPK), -1, dtype=torch.int32)
    fill = [0] * m
    order = torch.randperm(len(experts), generator=gen).tolist()
    for i, j in enumerate(order):  # one route each, round-robin over tokens
        t = i % m
        ids[t, fill[t]] = experts[j]
        fill[t] += 1
    w = torch.rand(len(experts), generator=gen) ** 3 + 0.05  # skewed popularity
    for t in range(m):
        have = set(ids[t, :fill[t]].tolist())
        while fill[t] < TOPK:
            j = int(torch.multinomial(w, 1, generator=gen))
            if experts[j] in have:
                continue
            ids[t, fill[t]] = experts[j]
            have.add(experts[j])
            fill[t] += 1
    return ids


def time_cells(variant, m, cells, numa):
    from vllm.model_executor.offloader.grace import GraceAllocation

    dev = torch.device("cuda:0")
    inter = inter_of(variant)
    ext = extension(variant)
    gen = torch.Generator().manual_seed(m)
    pool_hot = max(h for h, _ in cells) + 16
    pool_cold = max(c for _, c in cells) + 8

    def shapes(e):
        return {"w13_weight_packed": ((e, HIDDEN // 16, 2 * inter * 2), torch.int32),
                "w2_weight_packed": ((e, inter // 16, HIDDEN * 2), torch.int32),
                "w13_weight_scale": ((e, HIDDEN // 32, 2 * inter), torch.bfloat16),
                "w2_weight_scale": ((e, inter // 32, HIDDEN), torch.bfloat16)}

    hot = {k: torch.empty(s, dtype=d, device=dev) for k, (s, d) in shapes(pool_hot).items()}
    keep, cold = [], {}
    for k, (s, d) in shapes(pool_cold).items():
        a = GraceAllocation.allocate_pinned(s, d, 0, numa)
        keep.append(a)
        cold[k] = a.cuda_alias
    grid = json.loads((HERE.parent / "2026-10-08-m32/grid-m8-16-32.json").read_text())[str(m)]
    ntok = torch.tensor(grid["ntok"][1:m + 1], dtype=torch.float)
    x = (torch.randn((m, HIDDEN), generator=gen) * 0.3).to(torch.bfloat16).to(dev)
    for h, c in cells:
        calls = []
        for _ in range(20):
            ids = (all_routing(m, h, c, gen) if inter < 2048
                   else ep_routing(m, h, c, gen, ntok / ntok.sum()))
            hot_map = torch.full((512,), -1, dtype=torch.int32)
            cold_map = torch.full((512,), -1, dtype=torch.int32)
            hot_map[:h] = torch.randperm(pool_hot, generator=gen)[:h].to(torch.int32)
            cold_map[256:256 + c] = torch.randperm(pool_cold, generator=gen)[:c].to(torch.int32)
            calls.append([t.to(dev) for t in (ids, torch.rand((m, TOPK), generator=gen),
                                               hot_map, cold_map)])
        run = [lambda cl=cl: forward(ext, x, cl[0], cl[1], cl[2], cl[3], hot, cold)
               for cl in calls]
        for f in run[:2]:
            f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for f in run:
                f()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(10):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(5):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e) * 1000 / 100)
        print(json.dumps({"variant": variant, "m": m, "hot": h, "cold": c,
                          "us": round(best, 2)}), flush=True)
    del keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--m", type=int)
    ap.add_argument("--cells", nargs="+", default=[])
    ap.add_argument("--numa-node", type=int, default=0)
    a = ap.parse_args()
    if a.check:
        check()
    if a.time:
        time_cells(a.variant, a.m, [tuple(map(int, c.split(","))) for c in a.cells],
                   a.numa_node)


if __name__ == "__main__":
    sys.exit(main())
