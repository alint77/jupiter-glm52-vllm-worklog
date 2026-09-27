#!/usr/bin/env python3
"""route_prep with the in-kernel replica assignment vs with precomputed maps.

69 layers with the r1500 profile's placement (GPU 0's view) and one held-out
MiMo decode step's routes (8 tokens x top-8), one CUDA graph; see
bench_small.profile_graph. Slots are
handed out only to experts routed in that layer, so the small test tiers
(40 each) suffice.
"""

import json
import os
import sys
from pathlib import Path

import torch

import bench_small as bs
from bench_vllm_decode import HIDDEN, TOKENS, tier


def main() -> None:
    from vllm.model_executor.layers.fused_moe.tiered_decode import (
        Placement,
        tiered_decode_moe,
    )

    device = torch.device("cuda:0")
    prof = json.loads((bs.HERE.parents[1] / "profiles/mimo26-profile-3827-r1500.json").read_text())
    sys.path.insert(0, str(bs.HERE.parent / "2026-09-26-mimo-routing-profile"))
    from mimo_replicas import load_steps
    traces = Path(f"/e/fscratch/profound/{os.environ['USER']}/mimo26-route-cap/merged")
    step = int(os.environ.get("BENCH_STEP", "100"))
    routes = load_steps(traces, "heldout", [int(v) for v in prof["routed_layers"]])[step]
    gen = torch.Generator().manual_seed(0)
    hot_t, _ = tier(40, device, False, 1)
    cold_t, keep = tier(40, device, True, 1)
    x = (torch.randn((TOKENS, HIDDEN), generator=gen) * 0.3).to(torch.bfloat16).to(device)
    layers = []
    for li in range(bs.LAYERS):
        primary = torch.tensor(prof["owners"][li], dtype=torch.int32)
        secondary = torch.tensor(prof["secondary_ranks"][li], dtype=torch.int32)
        hot = torch.zeros(384, dtype=torch.int32)
        hot[prof["hot_experts"][li]] = 1
        topk = torch.from_numpy(routes[:, li, :]).int()
        active = torch.zeros(384, dtype=torch.bool)
        active[topk.flatten().long()] = True
        hot_mine = active & (hot > 0) & (primary == 0)
        cold_held = active & (hot == 0) & ((primary == 0) | (secondary == 0))
        hot_map = torch.where(hot_mine, torch.cumsum(hot_mine.int(), 0) - 1, -1).int()
        cold_map = torch.where(cold_held, torch.cumsum(cold_held.int(), 0) - 1, -1).int()
        assert hot_map.max() < 40 and cold_map.max() < 40
        weights = torch.rand((TOKENS, 8), generator=gen)
        placement = Placement(primary.to(device), secondary.to(device), hot.to(device), 0,
                              os.environ.get("BENCH_SCHEDULE", "1") == "1")
        layers.append((topk.to(device), weights.to(device), hot_map.to(device),
                       cold_map.to(device), placement))

    def run(with_placement: bool):
        def fn():
            for topk, w, hm, cm, pl in layers:
                tiered_decode_moe(x, topk, w, hm, cm, hot_t, cold_t, pl if with_placement else None)
        return fn

    bs.profile_graph(run(False), "precomputed maps")
    bs.profile_graph(run(True), "in-kernel assignment")


if __name__ == "__main__":
    main()
