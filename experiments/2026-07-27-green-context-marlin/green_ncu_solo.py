#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel resource footprint for hot vs cold Marlin (ncu attribution).

Runs ONE tier's Marlin gemm repeatedly so `ncu --kernel-name regex:Marlin` can
collect clean SpeedOfLight / MemoryWorkloadAnalysis / stall counters for that
kernel in isolation. From the hot and cold footprints we read off which shared
resource (DRAM/HBM vs L2 vs L1) each kernel stresses, and thus which would
saturate when both co-run (the ~+45% hot dilation seen in §4.5).

Usage:
  ncu --kernel-name regex:Marlin --launch-skip 30 --launch-count 4 \
      --section SpeedOfLight --section MemoryWorkloadAnalysis \
      --section SchedulerStats --section WarpStateStats \
      python green_ncu_solo.py hot
  ... and the same with `cold`.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

_MGP = "/e/project1/profound/alint77/vllm/agent_space/experiments/2026-07-27-green-context-marlin/marlin_green_probe.py"
_spec = importlib.util.spec_from_file_location("mgp", _MGP)
mgp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mgp)


def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else "hot"
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    device = torch.device("cuda:0")
    numa = mgp.detect_grace_numa_node(0)
    fh, fc = mgp.build_probe(16, 19, 3, cold_share=0.13, seed=13, device=device,
                             numa_node=numa, hbm_cold=False)
    fn = fh if tier == "hot" else fc
    torch.cuda.synchronize()
    for _ in range(60):  # warmup + steady-state launches for ncu to sample
        fn()
    torch.cuda.synchronize()
    print(f"profiled tier={tier}")


if __name__ == "__main__":
    main()
