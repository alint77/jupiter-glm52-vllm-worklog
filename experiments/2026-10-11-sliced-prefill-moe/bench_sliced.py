"""tiered_prefill_moe at the tp_sliced shapes: all 256 experts on the GPU, each
a 512-wide intermediate slice (w13 6144 -> 1024, w2 512 -> 6144), 191 hot +
65 cold (both HBM, as with the staged cold slot), real GLM-5.3 routing chunks
(every route is local). Per size: MoE wall time (CUDA events over 20 graph
replays, one launch each) and the per-phase kernel time from the torch
profiler, against the floors (630 TFLOPS bf16, 3.35 TB/s HBM).

    bench_sliced.py [--sizes 512,1024,2048,4096] [--samples 4] [--json out]
"""
import argparse
import re
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-02-ingraph-prefetch"))
import real_routing  # noqa: E402
from vllm.model_executor.layers.fused_moe import tiered_prefill  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (  # noqa: E402
    marlin_quantize,
)
from vllm.scalar_type import scalar_types  # noqa: E402

H, I, E, HOT = 6144, 512, 256, 191
Q = scalar_types.uint4b8
dev = torch.device("cuda", 0)


def tier(n, k_, f, seed):
    torch.manual_seed(seed)
    w = torch.randn((k_, f), dtype=torch.bfloat16, device=dev) / k_**0.5
    _, q, s, _, _, _ = marlin_quantize(w, Q, 32, False)
    return q[None].repeat(n, 1, 1).contiguous(), s[None].repeat(n, 1, 1).contiguous()


def comps(n, seed):
    w13, s13 = tier(n, H, 2 * I, seed)
    w2, s2 = tier(n, I, H, seed + 1)
    return {"w13_weight_packed": w13, "w13_weight_scale": s13,
            "w2_weight_packed": w2, "w2_weight_scale": s2}


def graph(fn):
    for _ in range(2):
        fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.accelerator.synchronize()
    return g


def wall_us(g, reps=20):
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.accelerator.synchronize()
    return a.elapsed_time(b) * 1000 / reps


def phases_us(g, reps=5, timeline=False):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.accelerator.synchronize()
    ev = sorted((e for e in prof.events() if e.device_type.name == "CUDA"),
                key=lambda e: e.time_range.start)
    out, after_act = {}, False
    if timeline:
        last = [e for e in ev if "route" in e.name][-1].time_range.start
        for e in ev:
            if e.time_range.start >= last:
                nm = re.search(r"(\w+_kernel)(<\D*(\d+))?", e.name)
                print(f"     {e.time_range.start - last:8.1f} {e.device_time:8.1f}"
                      f"  {nm[1]} {nm[3] or ''}")
    for e in ev:
        n = e.name
        if "route" in n:
            k, after_act = "route", False
        elif "gather" in n:
            k = "gather"
        elif "act_kernel" in n:
            k, after_act = "act", True
        elif "combine" in n:
            k = "combine"
        elif "gemm_kernel" in n:
            k = ("w2" if after_act else "w13") + " nt" + re.search(r"gemm_kernel<\D*(\d+)", n)[1]
        else:
            k = "other"
        out[k] = out.get(k, 0.0) + e.device_time / reps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,1024,2048,4096")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--json")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--schedule", type=int, default=0)
    a = ap.parse_args()
    hot, cold = comps(HOT, 0), comps(E - HOT, 10)
    k_exp = tiered_prefill.prefill_scale_exponent(hot, cold)
    rng = np.random.default_rng(0)
    perm = torch.from_numpy(rng.permutation(E).astype(np.int64))
    hm = torch.full((E,), -1, dtype=torch.int32)
    cm = torch.full((E,), -1, dtype=torch.int32)
    hm[perm[:HOT]] = torch.arange(HOT, dtype=torch.int32)
    cm[perm[HOT:]] = torch.arange(E - HOT, dtype=torch.int32)
    hm, cm = hm.to(dev), cm.to(dev)
    wbytes = sum(t.numel() * t.element_size() for c in (hot, cold) for t in c.values())
    results = []
    for chunk in [int(s) for s in a.sizes.split(",")]:
        walls, ph = [], {}
        smps = list(real_routing.samples(chunk, a.samples, seed=chunk))
        first = smps[0]
        for smp in smps:
            ids = torch.from_numpy(smp.topk_ids).to(dev)
            wts = torch.rand(ids.shape, device=dev).softmax(-1)
            x = torch.randn((chunk, H), dtype=torch.bfloat16, device=dev)
            g = graph(lambda: tiered_prefill.tiered_prefill_moe(
                x, ids, wts, hm, cm, hot, cold, k_exp, a.schedule))
            walls.append(wall_us(g))
            for k, v in phases_us(g, timeline=a.timeline and smp is first).items():
                ph[k] = ph.get(k, 0.0) + v / a.samples
        routes = chunk * 8
        flop = routes * 2 * (H * 2 * I + I * H)
        floor = max(flop / 630e12, wbytes / 3.35e12) * 1e6
        w = float(np.mean(walls))
        row = {"chunk": chunk, "wall_us": w, "floor_us": floor, "phases": ph}
        results.append(row)
        print(f"T={chunk}: wall {w:7.1f} us (floor {floor:.0f}: compute {flop / 630e12 * 1e6:.0f},"
              f" weights {wbytes / 3.35e12 * 1e6:.0f}; {w / floor:.2f}x)", flush=True)
        print("   " + "  ".join(f"{k} {v:.0f}" for k, v in sorted(ph.items())), flush=True)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
