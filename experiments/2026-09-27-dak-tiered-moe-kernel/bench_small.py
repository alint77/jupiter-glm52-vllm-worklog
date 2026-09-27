#!/usr/bin/env python3
"""The small per-layer MoE kernels, before and after trimming, on one GPU.

    bench_small.py assign            # replica assign with vs without Marlin align
    bench_small.py layer [--cu-dir D] # one-kernel layer, per-kernel durations

Each case is a CUDA graph of 69 calls (one decode step's MoE layers), replayed
under the torch profiler; kernel durations are profiler means, the wall clock
is graph replay per call. `--cu-dir` builds tiered_decode.cu from another
directory (e.g. the committed version) into its own cache.
"""

import argparse
import collections
import json
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LAYERS = 69


def profile_graph(fn, label: str) -> dict:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(10):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(5):
            graph.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / (5 * LAYERS))
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
    durs = collections.defaultdict(list)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            durs[ev.name].append(ev.device_time)
    kernels = {k[:70]: sum(v) / len(v) for k, v in durs.items() if len(v) >= 10 * LAYERS}
    out = {"case": label, "wall_us_per_call": best, "kernels_us": kernels}
    print(json.dumps(out), flush=True)
    return out


def bench_assign(device) -> None:
    from vllm.model_executor.model_loader.tiered_moe_scheduler import (
        allocate_fused_routing,
        tiered_moe_assign,
        tiered_moe_assign_align,
    )
    prof = json.loads((HERE.parents[1] / "profiles/mimo26-profile-3827-r1500.json").read_text())
    gen = torch.Generator().manual_seed(0)
    layers = []
    for li in range(LAYERS):
        owners = torch.tensor(prof["owners"][li], dtype=torch.int32)
        secondary = torch.tensor(prof["secondary_ranks"][li], dtype=torch.int32)
        hot = torch.zeros(owners.numel(), dtype=torch.int32)
        hot[prof["hot_experts"][li]] = 1
        mine = owners == 0
        hot_map = torch.where(mine & (hot > 0), torch.cumsum((mine & (hot > 0)).int(), 0) - 1, -1)
        holds = (mine & (hot == 0)) | (secondary == 0)
        cold_map = torch.where(holds, torch.cumsum(holds.int(), 0) - 1, -1)
        topk = torch.stack([torch.randperm(owners.numel(), generator=gen)[:8] for _ in range(8)])
        layers.append([t.to(device) for t in (topk.int(), owners, secondary, hot,
                                              hot_map.int(), cold_map.int())])
    bufs = [allocate_fused_routing(64, 384, 16, 16, device) for _ in range(LAYERS)]

    def align():
        for (topk, owners, secondary, hot, hmap, cmap), b in zip(layers, bufs):
            tiered_moe_assign_align(topk, owners, secondary, hot, hmap, cmap, *b, 0, 16, 16, True)

    def only():
        for (topk, owners, secondary, hot, hmap, cmap), b in zip(layers, bufs):
            tiered_moe_assign(topk, owners, secondary, hot, hmap, cmap, *b[:4], 0, True)

    profile_graph(align, "assign+align")
    profile_graph(only, "assign only")


def bench_layer(device, cu_dir: Path | None, cells: list[tuple[int, int]]) -> None:
    import vllm.model_executor.layers.fused_moe.tiered_decode as td
    if cu_dir is not None:
        td._HERE = cu_dir
    from bench_vllm_decode import HIDDEN, TOKENS, routing, tier
    gen = torch.Generator().manual_seed(0)
    hot_t, _ = tier(16, device, False, 1)
    cold_t, keep = tier(8, device, True, 1)
    x = (torch.randn((TOKENS, HIDDEN), generator=gen) * 0.3).to(torch.bfloat16).to(device)
    for hot, cold in cells:
        calls = [routing(hot, cold, 16, 8, r, gen, device) for r in range(LAYERS)]

        def run():
            for c in calls:
                td.tiered_decode_moe(x, c[0], c[1], c[2], c[3], hot_t, cold_t)

        profile_graph(run, f"layer h{hot} c{cold} {'head' if cu_dir else 'working tree'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["assign", "layer"])
    ap.add_argument("--cu-dir", type=Path)
    ap.add_argument("--cells", nargs="*", default=["9,2", "12,1", "4,3"])
    args = ap.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if args.what == "assign":
        bench_assign(device)
    else:
        cells = [tuple(map(int, c.split(","))) for c in args.cells]
        bench_layer(device, args.cu_dir, cells)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_TIERED_DECODE_PDL", "0")
    main()
