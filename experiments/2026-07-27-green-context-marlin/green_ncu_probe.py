#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware-counter attribution for the Track A hot-dilation mechanism.

The §4.5 sweep showed hot dilates ~+45% when cold co-runs under disjoint green
contexts, independent of the SM split and of cold's memory source. The README
attributed this to "L2 / memory fabric" — an inference, not a measurement. This
harness runs sustained, NVTX-marked phases so nsys --gpu-metrics (and ncu) can
identify the ACTUAL shared resource: HBM (DRAM) throughput, L2 throughput/hit,
C2C/NVLink, or output-write traffic.

Phases (each ~`--ms` long, separated by syncs so the metrics timeline segments
cleanly):
  A_hot_solo_green     hot on 116-SM green ctx
  B_cold_solo_green    cold on 16-SM green ctx (Grace)
  C_concurrent_green   hot(116) + cold(16) disjoint green (the dilated case)
  D_hot_solo_full      hot on the full-device default stream (Problem-4 control)
  E_concurrent_prod    hot + cold on production two streams (control)
  F_concurrent_green_hf  C but hot launched FIRST (Problem-5 launch order)

Run on Booster under:  nsys profile --gpu-metrics-device=0 -o rep.nsys-rep
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time

import torch

_MGP = "/e/project1/profound/alint77/vllm/agent_space/experiments/2026-07-27-green-context-marlin/marlin_green_probe.py"
_spec = importlib.util.spec_from_file_location("mgp", _MGP)
mgp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mgp)

nvtx = torch.cuda.nvtx


def run_solo_for(fn, stream, ms):
    with torch.cuda.stream(stream):
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) * 1000.0 < ms:
            fn()
    torch.cuda.synchronize()


def run_concurrent_for(fh, fc, hstream, cstream, ms, hot_first=False):
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1000.0 < ms:
        if hot_first:
            with torch.cuda.stream(hstream):
                fh()
            with torch.cuda.stream(cstream):
                fc()
        else:
            with torch.cuda.stream(cstream):
                fc()
            with torch.cuda.stream(hstream):
                fh()
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=800.0, help="per-phase duration")
    ap.add_argument("--cold-sm", type=int, default=16)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--hot-experts", type=int, default=19)
    ap.add_argument("--cold-experts", type=int, default=3)
    ap.add_argument("--numa-node", type=int, default=-1)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    device = torch.device("cuda:0")
    if args.numa_node < 0:
        args.numa_node = mgp.detect_grace_numa_node(0)
    print(f"=== green ncu/nsys attribution probe  cold={args.cold_sm}SM Grace NUMA {args.numa_node} ===")

    cold_h, hot_h, csm, hsm = mgp.make_green_streams(args.cold_sm)
    cstream = mgp.green_stream(cold_h)
    hstream = mgp.green_stream(hot_h)
    fh, fc = mgp.build_probe(args.m, args.hot_experts, args.cold_experts,
                             cold_share=0.13, seed=13, device=device,
                             numa_node=args.numa_node, hbm_cold=False)
    torch.cuda.synchronize()

    # warmup all paths once
    fh(); fc(); torch.cuda.synchronize()

    default = torch.cuda.current_stream()
    aux = torch.cuda.Stream()

    def phase(name, fn):
        nvtx.range_push(name)
        fn()
        nvtx.range_pop()
        torch.cuda.synchronize()
        time.sleep(0.05)  # gap so the metrics timeline shows phase boundaries

    phase("A_hot_solo_green", lambda: run_solo_for(fh, hstream, args.ms))
    phase("B_cold_solo_green", lambda: run_solo_for(fc, cstream, args.ms))
    phase("C_concurrent_green", lambda: run_concurrent_for(fh, fc, hstream, cstream, args.ms))
    phase("D_hot_solo_full", lambda: run_solo_for(fh, default, args.ms))
    phase("E_concurrent_prod", lambda: run_concurrent_for(fh, fc, default, aux, args.ms))
    phase("F_concurrent_green_hotfirst", lambda: run_concurrent_for(fh, fc, hstream, cstream, args.ms, hot_first=True))
    print("done")


if __name__ == "__main__":
    main()
