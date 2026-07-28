#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Track A §4.5 — Marlin hot(HBM)/cold(Grace-C2C) under disjoint green contexts.

Extends the proven Phase-21 Marlin-MoE harness
(benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py) with the green-context
streams validated in green_context_probe.cu. The make-or-break Track A test:
do real Marlin kernels — hot reading HBM, cold reading Grace over C2C — retain
clean concurrency under disjoint green contexts, or does the Phase-23 zero-sum
dilation reappear because the shared HBM port / L2 / power domain is the real
bottleneck (not SM scheduling)?

Requires a Booster node (NUMA-paired Grace memory); login-node C2C does not
transfer (per the worklog benchmark-on-booster rule).

Usage:
  python marlin_green_probe.py --cold-sm 16 --iters 50
  python marlin_green_probe.py --sweep   # 120/8, 112/16, 104/24, 96/32
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import traceback
from typing import Any

import torch

import vllm._custom_ops as ops  # noqa: F401  (loads _moe_C / marlin ext)

# Reuse the proven harness helpers. The benchmark file is importable as a module
# because it lives under benchmarks/ which is on the vLLM path when installed.
_BENCH = "/e/project1/profound/alint77/vllm/benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py"
import importlib.util
_spec = importlib.util.spec_from_file_location("p1b_bench", _BENCH)
p1b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1b)
make_tier = p1b.make_tier
build_gemm = p1b.build_gemm
global_routing = p1b.global_routing
tier_map = p1b.tier_map
align = p1b.align
time_call = p1b.time_call
time_both_with_events = p1b.time_both_with_events

# ---- CUDA driver API for green contexts ------------------------------------
# Resolve the driver + runtime libs by absolute path (bare names are not on the
# loader path under the JUPITER module env; torch links them but ctypes can't
# find them by soname).
_CUDART_CANDIDATES = [
    "/e/software/default/stages/2026/software/CUDA/13/lib64/libcudart.so",
    "libcudart.so",
]
_LIBCUDA_CANDIDATES = [
    "/lib64/libcuda.so",
    "/lib64/libcuda.so.1",
    "libcuda.so",
]


def _load(libs, tag):
    last = None
    for p in libs:
        try:
            return ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            last = e
    raise RuntimeError(f"could not load {tag}: {last}")


CUDA = _load(_LIBCUDA_CANDIDATES, "libcuda (driver)")
_RT = _load(_CUDART_CANDIDATES, "libcudart (runtime)")

CU_DEV_RESOURCE_TYPE_SM = 1
CU_STREAM_NON_BLOCKING = 0x1
CU_GREEN_CTX_DEFAULT_STREAM = 0x1
CUDA_SUCCESS = 0


class CUdevSmResource(ctypes.Structure):
    _fields_ = [("smCount", ctypes.c_uint), ("minSmPartitionSize", ctypes.c_uint), ("smCoscheduledAlignment", ctypes.c_uint)]

class CUdevResource(ctypes.Structure):
    # mirror layout: type + 92-byte pad + union (sm fits in 48-byte external)
    _fields_ = [("type", ctypes.c_int), ("_pad", ctypes.c_ubyte * 92), ("sm", CUdevSmResource), ("_over", ctypes.c_ubyte * 48)]

CUgreenCtx = ctypes.c_void_p
CUstream = ctypes.c_void_p


def _chk(r, tag):
    if r != CUDA_SUCCESS:
        raise RuntimeError(f"driver API {tag} failed: {r}")


def make_green_streams(cold_sm: int):
    """Query device, split SMs (cold_sm cold + remainder hot), return two
    (cudaStream_t-as-int) green-context streams + provisioned counts."""
    CUDA.cuInit(0)
    dev = ctypes.c_int(0)
    # Force the primary context active (the C++ probe did cudaFree(0));
    # green-context creation retains the primary context per cuda.h docs.
    import torch
    torch.cuda.set_device(0)
    torch.cuda.synchronize()

    full = CUdevResource()
    _chk(CUDA.cuDeviceGetDevResource(dev, ctypes.byref(full), CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource")
    print(f"  device SM: smCount={full.sm.smCount} minPart={full.sm.minSmPartitionSize} align={full.sm.smCoscheduledAlignment}")

    cold = CUdevResource()
    remain = CUdevResource()
    nb = ctypes.c_uint(1)
    _chk(CUDA.cuDevSmResourceSplitByCount(ctypes.byref(cold), ctypes.byref(nb), ctypes.byref(full), ctypes.byref(remain), 0, cold_sm), "cuDevSmResourceSplitByCount")

    cold_desc = ctypes.c_void_p()
    hot_desc = ctypes.c_void_p()
    _chk(CUDA.cuDevResourceGenerateDesc(ctypes.byref(cold_desc), ctypes.byref(cold), 1), "cuDevResourceGenerateDesc(cold)")
    _chk(CUDA.cuDevResourceGenerateDesc(ctypes.byref(hot_desc), ctypes.byref(remain), 1), "cuDevResourceGenerateDesc(hot)")

    cold_ctx = CUgreenCtx()
    hot_ctx = CUgreenCtx()
    _chk(CUDA.cuGreenCtxCreate(ctypes.byref(cold_ctx), cold_desc, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate(cold)")
    _chk(CUDA.cuGreenCtxCreate(ctypes.byref(hot_ctx), hot_desc, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate(hot)")

    cid = ctypes.c_ulonglong(0)
    hid = ctypes.c_ulonglong(0)
    CUDA.cuGreenCtxGetId(cold_ctx, ctypes.byref(cid))
    CUDA.cuGreenCtxGetId(hot_ctx, ctypes.byref(hid))

    least = ctypes.c_int(0)
    great = ctypes.c_int(0)
    _RT.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(great))

    cold_st = CUstream()
    hot_st = CUstream()
    _chk(CUDA.cuGreenCtxStreamCreate(ctypes.byref(cold_st), cold_ctx, CU_STREAM_NON_BLOCKING, great.value), "cuGreenCtxStreamCreate(cold)")
    _chk(CUDA.cuGreenCtxStreamCreate(ctypes.byref(hot_st), hot_ctx, CU_STREAM_NON_BLOCKING, great.value), "cuGreenCtxStreamCreate(hot)")

    # verify provisioned
    cg = CUdevResource()
    hg = CUdevResource()
    CUDA.cuGreenCtxGetDevResource(cold_ctx, ctypes.byref(cg), CU_DEV_RESOURCE_TYPE_SM)
    CUDA.cuGreenCtxGetDevResource(hot_ctx, ctypes.byref(hg), CU_DEV_RESOURCE_TYPE_SM)
    print(f"  provisioned: cold.smCount={cg.sm.smCount} hot.smCount={hg.sm.smCount} (sum={cg.sm.smCount+hg.sm.smCount}) greenCtxIds cold={cid.value} hot={hid.value}")

    return int(ctypes.cast(cold_st, ctypes.c_void_p).value or 0), int(ctypes.cast(hot_st, ctypes.c_void_p).value or 0), cg.sm.smCount, hg.sm.smCount


def green_stream(handle: int) -> torch.cuda.Stream:
    return torch.cuda.ExternalStream(handle)


def detect_grace_numa_node(device_index: int = 0) -> int:
    """Resolve the GPU-paired Grace NUMA node the way production does
    (gpu_model_runner -> get_device_numa_node -> nvmlDeviceGetNumaNodeId, with a
    CPU-affinity fallback for HBM-only NUMA nodes). The GPU<->Grace mapping is
    node-specific, so never hardcode it."""
    from vllm.platforms import current_platform
    node = current_platform.get_device_numa_node(device_index)
    if node is None:
        raise RuntimeError(f"could not detect Grace NUMA node for device {device_index}")
    return node


def build_probe(m: int, hot_e: int, cold_e: int, cold_share: float, seed: int, device, numa_node: int, hbm_cold: bool = False):
    """Construct hot (HBM) and cold (Marlin, HBM or Grace-UVA) tiers + gemm callables.

    Production model: ONE shared topk_ids over both tiers; each tier filters to
    its owned experts via its own expert_map (moe_align_block_size maps non-owned
    experts to -1). So build one combined routing with `cold_share` of the mass
    to cold, and align each tier against it with that tier's emap.

    hbm_cold=True places the cold tier in HBM too — a login-node-valid pessimistic
    test of whether two HBM-bound Marlin kernels contend under green contexts
    (shared HBM port). The real Booster run uses hbm_cold=False (cold -> Grace/C2C).
    """
    NUM = hot_e + cold_e
    device = torch.device(device) if not isinstance(device, torch.device) else device
    k = p1b.HIDDEN
    n = 2 * p1b.INTERMEDIATE  # w13 = gate+up; make_tier + build_gemm both use this n
    print(f"  make hot tier ({hot_e} experts, HBM)...")
    hq, hs = make_tier(hot_e, k, n, device, pinned=False)
    if hbm_cold:
        print(f"  make cold tier ({cold_e} experts, HBM [hbm_cold mode])...")
        cq, cs = make_tier(cold_e, k, n, device, pinned=False)
    else:
        print(f"  make cold tier ({cold_e} experts, Grace NUMA {numa_node})...")
        cq, cs = make_tier(cold_e, k, n, device, pinned=True, numa_node=numa_node)

    hot_ids = list(range(0, hot_e))
    cold_ids = list(range(hot_e, hot_e + cold_e))
    # one shared routing: cold_share of the topk mass lands on cold experts.
    # production: ~3 cold of ~22 activated -> cold_share ~0.13.
    if cold_share <= 0.0:
        cold_share = (cold_e / (hot_e + cold_e)) if (hot_e + cold_e) else 0.13
    topo = global_routing(m, hot_ids, cold_ids, NUM, device, seed, cold_share)

    hmap = tier_map(hot_ids, NUM, device)
    cmap = tier_map(cold_ids, NUM, device)

    a = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w = torch.ones((m, p1b.TOP_K), dtype=torch.float32, device=device)
    # Each tier needs its OWN Marlin workspace. The kernel's `locks` buffer is a
    # device-side spin-barrier for the cross-CTA reduction (marlin_template.h:
    # barrier_acquire spins `while (state != count)`). Two concurrent kernels
    # sharing one locks buffer corrupt each other's barrier counts -> infinite
    # spin = the 100%-GPU concurrent hang. Production uses separate workspace
    # views per tier (plan §1).
    ws_h = p1b.marlin_make_workspace_new(device, 4)  # max_blocks_per_sm=4 -> 528 >= min 384
    ws_c = p1b.marlin_make_workspace_new(device, 4)

    def mk(q, s, emap, ws):
        tok, eids, npost, blocks = align(topo, p1b.BLOCK_M, NUM, emap)
        out = torch.empty((m * p1b.TOP_K, n), dtype=torch.bfloat16, device=device)
        return build_gemm(a, out, q, s, ws, tok, eids, npost, weights=w,
                          m=m, n=n, k=k, cfg=(0, -1, -1))

    print(f"  align: hot blocks={align(topo, p1b.BLOCK_M, NUM, hmap)[3]}  cold blocks={align(topo, p1b.BLOCK_M, NUM, cmap)[3]}")
    fh = mk(hq, hs, hmap, ws_h)
    fc = mk(cq, cs, cmap, ws_c)
    return fh, fc


def time_on_stream(fn, stream, warmup=10, iters=50) -> float:
    """Solo timing with events recorded on the given (green) stream, so solo and
    concurrent (time_green_both, also per-stream) are measured consistently."""
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    with torch.cuda.stream(stream):
        s.record(stream)
        for _ in range(iters):
            fn()
        e.record(stream)
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # us


def time_green_both(fh, fc, hstream, cstream, warmup=10, iters=50) -> dict:
    """Concurrent hot(on hstream)+cold(on cstream) under disjoint green contexts.

    Launches both tiers INDEPENDENTLY with no cross-stream event waits. This
    mirrors production's overlap phase (hot on the main stream, cold on the aux
    stream; they join only downstream at the output add) and avoids the
    cross-green-context wait_stream() dependency that deadlocked the earlier
    fork/join version (isolated per-context workqueues + cross-context
    cudaStreamWaitEvent — the C++ mechanism probe never exercised that path, it
    launched both kernels independently and host-synced, which is why it passed).

    Union uses a single anchor event recorded before either launch; elapsed_time
    works across events on different streams, so
    union = max(anchor->hot_end, anchor->cold_end). Per-stream durations come
    from each stream's own start/end events.
    """
    def launch():
        with torch.cuda.stream(cstream):
            fc()
        with torch.cuda.stream(hstream):
            fh()

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    hot_ms = cold_ms = union_ms = 0.0
    for _ in range(iters):
        t0 = torch.cuda.Event(True)
        hs = torch.cuda.Event(True); he = torch.cuda.Event(True)
        cs = torch.cuda.Event(True); ce = torch.cuda.Event(True)
        t0.record(hstream)
        with torch.cuda.stream(cstream):
            cs.record(cstream); fc(); ce.record(cstream)
        with torch.cuda.stream(hstream):
            hs.record(hstream); fh(); he.record(hstream)
        torch.cuda.synchronize()
        hot_ms += hs.elapsed_time(he)
        cold_ms += cs.elapsed_time(ce)
        union_ms += max(t0.elapsed_time(he), t0.elapsed_time(ce))
    inv = 1000.0 / iters
    hot_us = hot_ms * inv
    cold_us = cold_ms * inv
    union_us = union_ms * inv
    serial_us = hot_us + cold_us
    overlap_us = max(0.0, serial_us - union_us)
    coresident = overlap_us / min(hot_us, cold_us) if min(hot_us, cold_us) > 0 else 0.0
    return {"hot_us": hot_us, "cold_us": cold_us, "union_us": union_us,
            "serial_us": serial_us, "coresident_frac": coresident}


def measure_clocks(launch_once, duration_ms=300, sample_s=0.005) -> dict:
    """Drive launch_once for ~duration_ms while a monitor thread polls SM clock,
    mem clock, and board power via NVML; return medians. Tests the plan §3 rule-7
    hypothesis: does concurrent execution drop SM clocks under the power cap (a
    wall no SM partition can fix)? NVML is context-independent, so polling from a
    side thread while CUDA runs is safe."""
    import statistics
    import threading
    import time

    from vllm.third_party import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    sm_l, mem_l, pw_l = [], [], []
    stop = threading.Event()

    def mon():
        while not stop.is_set():
            try:
                sm_l.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                mem_l.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
                pw_l.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
            except Exception:
                pass
            time.sleep(sample_s)

    for _ in range(8):
        launch_once()
    torch.cuda.synchronize()
    t = threading.Thread(target=mon, daemon=True)
    t.start()
    t_end = time.time() + duration_ms / 1000.0
    while time.time() < t_end:
        for _ in range(64):
            launch_once()
        torch.cuda.synchronize()
    stop.set()
    t.join()
    med = lambda v: statistics.median(v) if v else 0.0
    return {"sm_mhz": med(sm_l), "mem_mhz": med(mem_l),
            "power_w": med(pw_l), "nsamples": len(sm_l)}


def run_split(cold_sm: int, m: int, hot_e: int, cold_e: int, iters: int, device, numa_node, hbm_cold: bool = False):
    device = torch.device(device) if not isinstance(device, torch.device) else device
    print(f"\n=== split cold={cold_sm} SMs (hot={132-cold_sm}) ===")
    cold_h, hot_h, csm, hsm = make_green_streams(cold_sm)
    cstream = green_stream(cold_h)
    hstream = green_stream(hot_h)
    fh, fc = build_probe(m, hot_e, cold_e, cold_share=0.13, seed=13, device=device, numa_node=numa_node, hbm_cold=hbm_cold)
    torch.cuda.synchronize()

    # solo on green streams (events on each green stream for consistency with concurrent)
    t_hot_solo = time_on_stream(fh, hstream, warmup=10, iters=iters)
    t_cold_solo = time_on_stream(fc, cstream, warmup=10, iters=iters)
    print(f"  SOLO us: hot={t_hot_solo:.1f} cold={t_cold_solo:.1f}  serial_sum={t_hot_solo+t_cold_solo:.1f}")

    # concurrent under green contexts (both tiers on their green streams)
    r = time_green_both(fh, fc, hstream, cstream, warmup=10, iters=iters)
    union = r["union_us"]
    print(f"  CONCURRENT us: hot_ov={r['hot_us']:.1f} cold_ov={r['cold_us']:.1f} union={union:.1f} coresident={r['coresident_frac']:.2f}")
    hot_interf = r["hot_us"] / t_hot_solo - 1
    cold_interf = r["cold_us"] / t_cold_solo - 1
    print(f"  interference: hot={hot_interf:+.1%} cold={cold_interf:+.1%}  overlap_speedup={((t_hot_solo+t_cold_solo)/union):.2f}x")

    # same-job production control (plan §4.5 F): hot on main stream, cold on aux
    # stream with the apply_tiered fork/join.
    ctrl = torch.cuda.Stream()
    rp = time_both_with_events(fh, fc, cold_stream=ctrl, order="cold_first", warmup=10, iters=iters)
    print(f"  PROD control: hot={rp['hot_us']:.1f} cold={rp['cold_us']:.1f} union={rp['union_us']:.1f} "
          f"(hot_intf={rp['hot_us']/t_hot_solo-1:+.1%}) coresident={rp.get('coresident_frac', 0):.2f}")
    return {
        "cold_sm": cold_sm, "hot_sm": hsm, "hbm_cold": hbm_cold,
        "hot_solo_us": t_hot_solo, "cold_solo_us": t_cold_solo,
        **r,
        "hot_interference": hot_interf, "cold_interference": cold_interf,
        "overlap_speedup_x": (t_hot_solo + t_cold_solo) / union,
        "prod": rp,
        "prod_hot_interference": rp["hot_us"] / t_hot_solo - 1,
        "green_vs_prod_union_frac": union / rp["union_us"] - 1,
    }


def run_diag(cold_sm: int, m: int, hot_e: int, cold_e: int, iters: int, device, numa_node, hbm_cold: bool = False):
    """Mechanism diagnosis for one split (plan §3 rules 2,7 + §4.5 control F).

    Adds to the sweep: (1) a plain-stream concurrent control (no SM isolation) to
    tell whether green contexts help/hurt vs the stock two-stream path, and
    (2) NVML SM/mem clock + board power during solo vs concurrent, to test the
    power/clock-wall hypothesis for the split-independent hot dilation.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    print(f"\n=== DIAG cold={cold_sm} SMs (hot={132-cold_sm}) ===")
    cold_h, hot_h, csm, hsm = make_green_streams(cold_sm)
    cstream = green_stream(cold_h)
    hstream = green_stream(hot_h)
    fh, fc = build_probe(m, hot_e, cold_e, cold_share=0.13, seed=13, device=device, numa_node=numa_node, hbm_cold=hbm_cold)
    torch.cuda.synchronize()

    def hot_only():
        with torch.cuda.stream(hstream):
            fh()

    def cold_only():
        with torch.cuda.stream(cstream):
            fc()

    def both():
        with torch.cuda.stream(cstream):
            fc()
        with torch.cuda.stream(hstream):
            fh()

    t_hot_solo = time_on_stream(fh, hstream, warmup=10, iters=iters)
    t_cold_solo = time_on_stream(fc, cstream, warmup=10, iters=iters)
    r_green = time_green_both(fh, fc, hstream, cstream, warmup=10, iters=iters)

    # Control (plan §4.5 F): the production concurrent path — hot on the main
    # stream, cold on an aux stream, with the apply_tiered fork/join
    # (cold.wait(main) -> both -> main.wait(cold)). This co-runs the tiers the
    # way production does, so it is the real bar the green-context variant must
    # beat. Also keep a plain two-stream fork (no isolation, no join) to show the
    # scheduler-serialization floor.
    ctrl_stream = torch.cuda.Stream()
    r_prod = time_both_with_events(fh, fc, cold_stream=ctrl_stream, order="cold_first", warmup=10, iters=iters)
    plain_h = torch.cuda.Stream()
    plain_c = torch.cuda.Stream()
    r_plain = time_green_both(fh, fc, plain_h, plain_c, warmup=10, iters=iters)

    print(f"  SOLO us:        hot={t_hot_solo:.1f} cold={t_cold_solo:.1f}")
    print(f"  GREEN  concur:  hot={r_green['hot_us']:.1f} cold={r_green['cold_us']:.1f} union={r_green['union_us']:.1f} "
          f"(hot_intf={r_green['hot_us']/t_hot_solo-1:+.1%})")
    print(f"  PROD   concur:  hot={r_prod['hot_us']:.1f} cold={r_prod['cold_us']:.1f} union={r_prod['union_us']:.1f} "
          f"(hot_intf={r_prod['hot_us']/t_hot_solo-1:+.1%}) coresident={r_prod.get('coresident_frac', 0):.2f}  [production control]")
    print(f"  PLAIN  concur:  hot={r_plain['hot_us']:.1f} cold={r_plain['cold_us']:.1f} union={r_plain['union_us']:.1f} "
          f"(hot_intf={r_plain['hot_us']/t_hot_solo-1:+.1%})  [no isolation, no join]")

    clk_hot = measure_clocks(hot_only)
    clk_cold = measure_clocks(cold_only)
    clk_both = measure_clocks(both)
    print(f"  CLOCKS hot-solo:    SM={clk_hot['sm_mhz']:.0f}MHz mem={clk_hot['mem_mhz']:.0f}MHz pow={clk_hot['power_w']:.0f}W (n={clk_hot['nsamples']})")
    print(f"  CLOCKS cold-solo:   SM={clk_cold['sm_mhz']:.0f}MHz mem={clk_cold['mem_mhz']:.0f}MHz pow={clk_cold['power_w']:.0f}W (n={clk_cold['nsamples']})")
    print(f"  CLOCKS concurrent:  SM={clk_both['sm_mhz']:.0f}MHz mem={clk_both['mem_mhz']:.0f}MHz pow={clk_both['power_w']:.0f}W (n={clk_both['nsamples']})")
    sm_drop = clk_both['sm_mhz'] / clk_hot['sm_mhz'] - 1 if clk_hot['sm_mhz'] else 0.0
    print(f"  concurrent SM-clock vs hot-solo: {sm_drop:+.1%}")
    return {
        "cold_sm": cold_sm, "hot_sm": hsm, "hbm_cold": hbm_cold,
        "hot_solo_us": t_hot_solo, "cold_solo_us": t_cold_solo,
        "green": r_green, "prod": r_prod, "plain": r_plain,
        "clocks": {"hot_solo": clk_hot, "cold_solo": clk_cold, "concurrent": clk_both},
        "sm_clock_drop_concurrent_vs_hot_solo": sm_drop,
    }


def run_stage(m: int, hot_e: int, cold_e: int, iters: int, device, numa_node):
    """Feasibility of Grace->HBM cold-weight staging (the user's pivot after the
    FAIL-A verdict): overlap the C2C weight transfer with hot Marlin, then run
    cold Marlin from the staged HBM copy AFTER hot retires (no co-residency).

    Measures the four unknowns that decide it:
      1. transfer solo time (does it fit under hot's ~132us?).
      2. transfer CONCURRENT with hot -> does the C2C copy steal hot's HBM
         bandwidth (the co-residency question, but for a copy not a kernel)?
      3. cold Marlin from staged HBM (the phase-2 cost).
      4. end-to-end staged per-layer latency vs the production co-run union.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    NUM = hot_e + cold_e
    k = p1b.HIDDEN
    n = 2 * p1b.INTERMEDIATE
    print(f"\n=== STAGE M={m} hot={hot_e} cold={cold_e} (Grace->HBM staging) ===")
    hq, hs = make_tier(hot_e, k, n, device, pinned=False)
    print(f"  make cold tier ({cold_e} experts, Grace NUMA {numa_node})...")
    cq_g, cs_g = make_tier(cold_e, k, n, device, pinned=True, numa_node=numa_node)

    hot_ids = list(range(hot_e)); cold_ids = list(range(hot_e, NUM))
    topo = global_routing(m, hot_ids, cold_ids, NUM, device, 13, 0.13)
    hmap = tier_map(hot_ids, NUM, device); cmap = tier_map(cold_ids, NUM, device)
    a = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w = torch.ones((m, p1b.TOP_K), dtype=torch.float32, device=device)
    ws_h = p1b.marlin_make_workspace_new(device, 4)
    ws_cg = p1b.marlin_make_workspace_new(device, 4)
    ws_ch = p1b.marlin_make_workspace_new(device, 4)

    def mk(q, s, emap, ws):
        tok, eids, npost, blocks = align(topo, p1b.BLOCK_M, NUM, emap)
        out = torch.empty((m * p1b.TOP_K, n), dtype=torch.bfloat16, device=device)
        return build_gemm(a, out, q, s, ws, tok, eids, npost, weights=w,
                          m=m, n=n, k=k, cfg=(0, -1, -1)), out

    fh, out_h = mk(hq, hs, hmap, ws_h)              # hot Marlin (HBM)
    fc_grace, out_cg = mk(cq_g, cs_g, cmap, ws_cg)  # cold Marlin direct from Grace (prod)

    # HBM staging buffers + the transfer op (Grace-UVA -> HBM, over C2C)
    cq_h = torch.empty_like(cq_g); cs_h = torch.empty_like(cs_g)
    nbytes = cq_g.numel() * cq_g.element_size() + cs_g.numel() * cs_g.element_size()

    def transfer():
        cq_h.copy_(cq_g, non_blocking=True)
        cs_h.copy_(cs_g, non_blocking=True)

    fc_hbm, out_ch = mk(cq_h, cs_h, cmap, ws_ch)     # cold Marlin from staged HBM

    # Correctness (gate: bit-exact kernel output). Staged copy of the same packed
    # weights through the same kernel must reproduce the direct-from-Grace output.
    # Zero-init both outs so rows the kernel does not write compare equal (the
    # empty-init garbage in unwritten rows otherwise reads as a spurious NaN).
    out_cg.zero_()
    out_ch.zero_()
    transfer()
    torch.cuda.synchronize()
    fc_grace()
    torch.cuda.synchronize()
    fc_hbm()
    torch.cuda.synchronize()
    exact = torch.equal(out_cg, out_ch)
    d = (out_cg.float() - out_ch.float()).abs()
    dfin = d[torch.isfinite(d)]
    maxdiff = dfin.max().item() if dfin.numel() else float("nan")
    # Determinism control: does the direct-from-Grace kernel reproduce ITSELF at
    # this m? If direct-vs-direct shows the same ~1e-3 spread, the staged diff is
    # kernel reduction-order nondeterminism at high block counts, not a staging bug.
    snap = out_cg.clone()
    out_cg.zero_()
    torch.cuda.synchronize()
    fc_grace()
    torch.cuda.synchronize()
    sd = (snap.float() - out_cg.float()).abs()
    sdfin = sd[torch.isfinite(sd)]
    selfdiff = sdfin.max().item() if sdfin.numel() else float("nan")
    # Staged self-determinism: production would run the staged path consistently,
    # so what matters is that staged reproduces itself (and is within reduction-
    # order rounding of direct), not that it bit-matches the Grace path.
    snap_h = out_ch.clone()
    out_ch.zero_()
    torch.cuda.synchronize()
    fc_hbm()
    torch.cuda.synchronize()
    sh = (snap_h.float() - out_ch.float()).abs()
    shfin = sh[torch.isfinite(sh)]
    selfdiff_h = shfin.max().item() if shfin.numel() else float("nan")
    print(f"  CORRECTNESS staged==direct: bitexact={exact} "
          f"nan_direct={int(torch.isnan(out_cg).sum())} nan_staged={int(torch.isnan(out_ch).sum())} "
          f"maxabsdiff_finite={maxdiff:.3e}  direct_vs_direct={selfdiff:.3e} staged_vs_staged={selfdiff_h:.3e}")

    main = torch.cuda.current_stream()
    aux = torch.cuda.Stream()

    # 1. solos
    t_hot = time_on_stream(fh, main, warmup=10, iters=iters)
    t_trans = time_on_stream(transfer, aux, warmup=10, iters=iters)
    transfer()  # ensure staged data is valid before fc_hbm timing
    torch.cuda.synchronize()
    t_cold_hbm = time_on_stream(fc_hbm, main, warmup=10, iters=iters)
    t_cold_grace = time_on_stream(fc_grace, main, warmup=10, iters=iters)
    c2c_gbs = nbytes / (t_trans * 1e-6) / 1e9 if t_trans > 0 else 0.0
    print(f"  SOLO us: hot={t_hot:.1f} transfer={t_trans:.1f} ({nbytes/1e6:.1f}MB -> {c2c_gbs:.0f}GB/s) "
          f"cold_from_HBM={t_cold_hbm:.1f} cold_from_Grace={t_cold_grace:.1f}")

    # 2. transfer CONCURRENT with hot: does the C2C copy dilate hot?
    r_th = time_green_both(fh, transfer, main, aux, warmup=10, iters=iters)
    hot_dil = r_th["hot_us"] / t_hot - 1
    print(f"  TRANSFER+HOT concurrent: hot={r_th['hot_us']:.1f} (dil={hot_dil:+.1%}) "
          f"transfer={r_th['cold_us']:.1f} union={r_th['union_us']:.1f}")

    # 3. baseline: production co-run (hot + cold-direct-from-Grace)
    r_prod = time_both_with_events(fh, fc_grace, cold_stream=torch.cuda.Stream(), order="cold_first", warmup=10, iters=iters)
    print(f"  PROD co-run union={r_prod['union_us']:.1f} (hot={r_prod['hot_us']:.1f} cold={r_prod['cold_us']:.1f})")

    # 4. end-to-end staged per-layer latency: {transfer || hot} then cold_from_HBM
    def staged_step():
        with torch.cuda.stream(aux):
            transfer()
        with torch.cuda.stream(main):
            fh()
        main.wait_stream(aux)          # cold needs the staged weights
        with torch.cuda.stream(main):
            fc_hbm()

    for _ in range(10):
        staged_step()
    torch.cuda.synchronize()
    tot = 0.0
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(main); staged_step(); e.record(main)
        torch.cuda.synchronize()
        tot += s.elapsed_time(e)
    t_staged = tot * 1000.0 / iters
    est = max(t_hot, r_th["cold_us"]) + t_cold_hbm
    print(f"  STAGED per-layer: measured={t_staged:.1f}us  est(max(hot,transfer)+cold_HBM)={est:.1f}us")
    print(f"  vs PROD co-run {r_prod['union_us']:.1f}us -> staged is {t_staged/r_prod['union_us']-1:+.1%} "
          f"({'BETTER' if t_staged < r_prod['union_us'] else 'WORSE'})")
    return {
        "hot_solo_us": t_hot, "transfer_solo_us": t_trans, "transfer_mb": nbytes / 1e6,
        "c2c_gbs": c2c_gbs, "cold_from_hbm_us": t_cold_hbm, "cold_from_grace_us": t_cold_grace,
        "transfer_hot_concurrent": r_th, "hot_dilation_under_transfer": hot_dil,
        "prod_union_us": r_prod["union_us"], "staged_measured_us": t_staged, "staged_est_us": est,
        "staged_vs_prod_frac": t_staged / r_prod["union_us"] - 1,
        "correct_bitexact": exact, "correct_maxabsdiff": maxdiff,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold-sm", type=int, default=16)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="run one split with plain-stream control + NVML clock/power diagnosis")
    ap.add_argument("--stage", action="store_true",
                    help="test Grace->HBM cold-weight staging (transfer || hot, then cold from HBM)")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--hot-experts", type=int, default=19)
    ap.add_argument("--cold-experts", type=int, default=3)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--numa-node", type=int, default=-1,
                    help="GPU-paired Grace NUMA node; -1 = auto-detect via nvml (production path)")
    ap.add_argument("--hbm-cold", action="store_true",
                    help="place cold tier in HBM too (login-node pessimistic test; no Grace/C2C)")
    ap.add_argument("--out", default="results-marlin-green.json")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    device = torch.device("cuda:0")
    mode = "HBM-cold (pessimistic, login-valid)" if args.hbm_cold else "Grace/C2C cold (Booster)"
    print(f"=== Marlin green-context probe (GH200) M={args.m} hot={args.hot_experts} cold={args.cold_experts}  [{mode}] ===")

    if args.numa_node < 0 and not args.hbm_cold:
        args.numa_node = detect_grace_numa_node(0)
        print(f"  auto-detected GPU0-paired Grace NUMA node: {args.numa_node}")

    if args.stage:
        r = run_stage(args.m, args.hot_experts, args.cold_experts, args.iters, device, args.numa_node)
        json.dump({"args": vars(args), "result": r}, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
        return

    splits = [8, 16, 24, 32] if args.sweep else [args.cold_sm]
    runner = run_diag if args.diag else run_split
    out = []
    for cs in splits:
        try:
            out.append(runner(cs, args.m, args.hot_experts, args.cold_experts, args.iters, device, args.numa_node, hbm_cold=args.hbm_cold))
        except Exception:
            traceback.print_exc()
            out.append({"cold_sm": cs, "error": traceback.format_exc()})
    json.dump({"args": vars(args), "results": out}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    # summary
    if not args.diag:
        for r in out:
            if "error" not in r:
                print(f"  cold={r['cold_sm']:2d}SM  interf hot={r['hot_interference']:+.0%} cold={r['cold_interference']:+.0%}  speedup={r['overlap_speedup_x']:.2f}x  union={r['union_us']:.0f}us")


if __name__ == "__main__":
    sys.exit(main())
