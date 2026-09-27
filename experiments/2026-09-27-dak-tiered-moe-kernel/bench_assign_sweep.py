#!/usr/bin/env python3
"""Assign-only kernel: warps and schedule on/off, 69-layer graph (bench_small)."""

import sys

import torch

import bench_small as bs
from vllm.model_executor.model_loader import tiered_moe_scheduler as sched


def main() -> None:
    device = torch.device("cuda:0")
    captured = {}
    real = sched.tiered_moe_assign

    def grab(*a):
        captured.setdefault("calls", []).append(a)
    sched.tiered_moe_assign = grab
    bs.profile_graph.__globals__["torch"] = torch
    # collect the 69 layers' inputs through bench_assign's closure
    orig = bs.profile_graph
    bs.profile_graph = lambda fn, label: fn() if label == "assign only" else None
    bs.bench_assign.__globals__["tiered_moe_assign"] = grab
    import vllm.model_executor.model_loader.tiered_moe_scheduler as m
    m.tiered_moe_assign = grab
    bs.bench_assign(device)
    bs.profile_graph = orig
    calls = captured["calls"]
    for warps in (2, 4, 8):
        for schedule in (True, False):
            def run():
                for (topk, owners, secondary, hot, hmap, cmap, *b, rank, _s) in calls:
                    out = sched.FusedRouting(*b, *(b[0],) * 6)
                    sched._launch_assign_align(topk, owners, secondary, hot, sched.TierMaps(hmap, cmap),
                                               rank, 16, 16, schedule, out, num_warps=warps, align=False)
            orig(run, f"warps={warps} schedule={schedule}")


if __name__ == "__main__":
    sys.exit(main())
