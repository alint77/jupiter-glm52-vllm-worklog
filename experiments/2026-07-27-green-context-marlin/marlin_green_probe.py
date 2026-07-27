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
    ws = p1b.marlin_make_workspace_new(device, 4)  # max_blocks_per_sm=4 -> 528 >= min 384

    def mk(q, s, emap):
        tok, eids, npost, blocks = align(topo, p1b.BLOCK_M, NUM, emap)
        out = torch.empty((m * p1b.TOP_K, n), dtype=torch.bfloat16, device=device)
        return build_gemm(a, out, q, s, ws, tok, eids, npost, weights=w,
                          m=m, n=n, k=k, cfg=(0, -1, -1))

    print(f"  align: hot blocks={align(topo, p1b.BLOCK_M, NUM, hmap)[3]}  cold blocks={align(topo, p1b.BLOCK_M, NUM, cmap)[3]}")
    fh = mk(hq, hs, hmap)
    fc = mk(cq, cs, cmap)
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
    """Concurrent hot(on hstream)+cold(on cstream) under disjoint green contexts,
    mirroring production apply_tiered fork/join but with BOTH tiers on green
    streams (the harness's time_both_with_events puts hot on the default stream,
    which is wrong for the green-context test).

    Returns per-stream durations + union (wall) in us, plus co-residency.
    """
    def both():
        # fork: cold waits for hot's start point; both then run concurrently
        cstream.wait_stream(hstream)
        cs = torch.cuda.Event(True); ce = torch.cuda.Event(True)
        with torch.cuda.stream(cstream):
            cs.record(cstream)
            fc()
            ce.record(cstream)
        hs = torch.cuda.Event(True); he = torch.cuda.Event(True)
        with torch.cuda.stream(hstream):
            hs.record(hstream)
            fh()
            he.record(hstream)
        hstream.wait_stream(cstream)  # join
        return hs, he, cs, ce

    for _ in range(warmup):
        both()
    torch.cuda.synchronize()

    # per-stream durations
    hot_ms = 0.0; cold_ms = 0.0
    for _ in range(iters):
        hs, he, cs, ce = both()
        torch.cuda.synchronize()
        hot_ms += hs.elapsed_time(he)
        cold_ms += cs.elapsed_time(ce)
    # union (wall) via outer events on hstream around the whole both()
    u0 = torch.cuda.Event(True); u1 = torch.cuda.Event(True)
    for _ in range(warmup):
        both()
    torch.cuda.synchronize()
    utot = 0.0
    for _ in range(iters):
        u0.record(hstream)
        both()
        u1.record(hstream)
        torch.cuda.synchronize()
        utot += u0.elapsed_time(u1)
    inv = 1000.0 / iters
    hot_us = hot_ms * inv
    cold_us = cold_ms * inv
    union_us = utot * inv
    serial_us = hot_us + cold_us
    overlap_us = max(0.0, serial_us - union_us)
    coresident = overlap_us / min(hot_us, cold_us) if min(hot_us, cold_us) > 0 else 0.0
    return {"hot_us": hot_us, "cold_us": cold_us, "union_us": union_us,
            "serial_us": serial_us, "coresident_frac": coresident}


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
    return {
        "cold_sm": cold_sm, "hot_sm": hsm, "hbm_cold": hbm_cold,
        "hot_solo_us": t_hot_solo, "cold_solo_us": t_cold_solo,
        **r,
        "hot_interference": hot_interf, "cold_interference": cold_interf,
        "overlap_speedup_x": (t_hot_solo + t_cold_solo) / union,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold-sm", type=int, default=16)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--hot-experts", type=int, default=19)
    ap.add_argument("--cold-experts", type=int, default=3)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--numa-node", type=int, default=2)  # GPU0-paired Grace node
    ap.add_argument("--hbm-cold", action="store_true",
                    help="place cold tier in HBM too (login-node pessimistic test; no Grace/C2C)")
    ap.add_argument("--out", default="results-marlin-green.json")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    device = torch.device("cuda:0")
    mode = "HBM-cold (pessimistic, login-valid)" if args.hbm_cold else "Grace/C2C cold (Booster)"
    print(f"=== Marlin green-context probe (GH200) M={args.m} hot={args.hot_experts} cold={args.cold_experts}  [{mode}] ===")

    splits = [8, 16, 24, 32] if args.sweep else [args.cold_sm]
    out = []
    for cs in splits:
        try:
            out.append(run_split(cs, args.m, args.hot_experts, args.cold_experts, args.iters, device, args.numa_node, hbm_cold=args.hbm_cold))
        except Exception:
            traceback.print_exc()
            out.append({"cold_sm": cs, "error": traceback.format_exc()})
    json.dump({"args": vars(args), "results": out}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    # summary
    for r in out:
        if "error" not in r:
            print(f"  cold={r['cold_sm']:2d}SM  interf hot={r['hot_interference']:+.0%} cold={r['cold_interference']:+.0%}  speedup={r['overlap_speedup_x']:.2f}x  union={r['union_us']:.0f}us")


if __name__ == "__main__":
    sys.exit(main())
