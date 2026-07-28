#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Realistic full-chain (W13+W2) Grace->HBM staging probe for the q4 regime.

Answers the follow-up to the W13-only feasibility result: does Grace->HBM
cold-weight staging still help at the REAL operating point — the dominant q4
decode shape (m~4 tokens/rank, ~5 hot + ~1 cold experts/rank, ~20% cold) — and
where is the crossover as (m, hot, cold) vary?

Faithfulness fixes over the W13-only probe (per review):
  * full W13+W2 chain per tier (not just W13);
  * non-contiguous per-expert staging (gather each activated cold expert's
    W13+W2 slices from a Grace pool into fixed HBM slots), not one big copy;
  * one common-start / final-join timing structure shared by the staged and the
    direct-Grace controls (no cross-stream-wait fragility, no submission skew);
  * L2 flush between timed iterations so cold Grace reads are not L2-cached
    (matches production, where 74 other layers' weights evict L2 between reads);
  * the analytical estimate uses the CONCURRENT hot duration, not solo.

Chain note: per tier, chain = W13 gemm + W2 gemm run sequentially. The SiLU-mul
activation between them is tier-independent compute (identical for staged and
direct) so it cancels in the comparison; it is omitted to avoid the fragile
W13-output -> W2-input row-alignment dependency. Both gemms stream their real
weights, which is what the staging decision depends on.

Run on a Booster node (Grace/C2C + real Marlin). Login node is not valid for
C2C perf (benchmark-on-booster rule).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback

import torch

import vllm._custom_ops as ops  # noqa: F401  (loads marlin ext)

_BENCH = "/e/project1/profound/alint77/vllm/benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py"
_spec = importlib.util.spec_from_file_location("p1b_bench", _BENCH)
p1b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1b)
make_tier = p1b.make_tier
build_gemm = p1b.build_gemm
global_routing = p1b.global_routing
tier_map = p1b.tier_map
align = p1b.align

HIDDEN = p1b.HIDDEN            # 6144
INTERMEDIATE = p1b.INTERMEDIATE  # 2048
TOP_K = p1b.TOP_K              # 8
BLOCK_M = p1b.BLOCK_M          # 16


def balanced_routing(m, hot_ids, cold_ids, device, cold_share):
    """Deterministic, balanced topk_ids: round-robin so each hot expert gets
    equal token mass and each cold expert equal mass, split by cold_share. This
    isolates the (m, hot, cold) effect from per-cell routing noise (the harness's
    global_routing randomly picks WHICH experts get tokens, so cells were not
    comparable). cold=0 -> all mass to hot (no randint crash)."""
    import numpy as np
    n = m * TOP_K
    n_cold = int(round(n * cold_share)) if cold_ids else 0
    picks = [hot_ids[i % len(hot_ids)] for i in range(n - n_cold)]
    picks += [cold_ids[i % len(cold_ids)] for i in range(n_cold)]
    np.random.default_rng(0).shuffle(picks)
    return torch.tensor(picks, dtype=torch.int32, device=device).view(m, TOP_K)


def detect_grace_numa_node(device_index: int = 0) -> int:
    from vllm.platforms import current_platform
    node = current_platform.get_device_numa_node(device_index)
    if node is None:
        raise RuntimeError("could not detect Grace NUMA node")
    return node


def build_chain(a13, a2, w13q, w13s, w2q, w2s, emap, topo, ws13, ws2, m, device):
    """One tier's W13+W2 chain as a single callable (weight-streaming faithful).

    Both gemms share the same routing/alignment (same m, top_k=TOP_K, same
    tok/eids) so they cover the same m*TOP_K (token,expert) assignments; they
    differ only in the weight matrix and k/n. W13: (m,HIDDEN)->(m*TOP_K,
    2*INTERMEDIATE). W2: (m,INTERMEDIATE)->(m*TOP_K,HIDDEN). Weight streaming is
    identical to the real chain; the activation between them is tier-independent
    and cancels in staged-vs-direct, so it is omitted.
    """
    NUM = int(emap.numel())
    n13 = 2 * INTERMEDIATE
    tok, eids, npost, _ = align(topo, BLOCK_M, NUM, emap)
    w = torch.ones((m, TOP_K), dtype=torch.float32, device=device)
    out13 = torch.empty((m * TOP_K, n13), dtype=torch.bfloat16, device=device)
    g13 = build_gemm(a13, out13, w13q, w13s, ws13, tok, eids, npost,
                     weights=w, m=m, n=n13, k=HIDDEN, cfg=(0, -1, -1))
    out2 = torch.empty((m * TOP_K, HIDDEN), dtype=torch.bfloat16, device=device)
    g2 = build_gemm(a2, out2, w2q, w2s, ws2, tok, eids, npost,
                    weights=w, m=m, n=HIDDEN, k=INTERMEDIATE, cfg=(0, -1, -1))

    def chain():
        g13()
        g2()

    return chain, out2


class L2Flusher:
    """Defeats L2 caching of cold Grace reads between timed iters."""

    def __init__(self, device, mb=256):
        self.buf = torch.empty(mb * 1024 * 1024, dtype=torch.uint8, device=device)

    def flush(self):
        self.buf.zero_()


def time_common_start(fork_fn, join_stream, l2, warmup=8, iters=30):
    """Time fork_fn with a rigorous common start: fork_fn records its own start
    event on join_stream first, and all branch streams wait on it. Returns the
    median common-start->final-join duration in us, with an L2 flush each iter.
    """
    for _ in range(warmup):
        fork_fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        l2.flush()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record(join_stream)
        fork_fn(start_event=s)
        e.record(join_stream)
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def run_cell(m, n_hot, n_cold, device, numa_node, l2, iters, cold_pool=4):
    """Staged vs direct-Grace for one (m, n_hot, n_cold) cell."""
    device = torch.device(device)
    NUM = n_hot + cold_pool  # global expert ids: [0,n_hot) hot, [n_hot,NUM) cold pool
    k = HIDDEN
    n13 = 2 * INTERMEDIATE

    # Hot tier (W13+W2) in HBM.
    hq13, hs13 = make_tier(n_hot, HIDDEN, n13, device, pinned=False)
    hq2, hs2 = make_tier(n_hot, INTERMEDIATE, HIDDEN, device, pinned=False)

    # Cold pool (W13+W2) in Grace. n_cold activated experts drawn from it.
    cq13, cs13 = make_tier(cold_pool, HIDDEN, n13, device, pinned=True, numa_node=numa_node)
    cq2, cs2 = make_tier(cold_pool, INTERMEDIATE, HIDDEN, device, pinned=True, numa_node=numa_node)

    hot_ids = list(range(n_hot))
    cold_ids = list(range(n_hot, n_hot + n_cold))  # activated cold experts (first n_cold of pool)
    cold_share = (n_cold / (n_hot + n_cold)) if (n_hot + n_cold) else 0.0
    topo = balanced_routing(m, hot_ids, cold_ids, device, cold_share)
    a13 = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    a2 = torch.randn((m, INTERMEDIATE), dtype=torch.bfloat16, device=device)

    hmap = tier_map(hot_ids, NUM, device)
    cmap = (tier_map(cold_ids, NUM, device) if n_cold
            else torch.full((NUM,), -1, dtype=torch.int32, device=device))

    ws = lambda: p1b.marlin_make_workspace_new(device, 4)
    hot_chain, _ = build_chain(a13, a2, hq13, hs13, hq2, hs2, hmap, topo, ws(), ws(), m, device)

    # DIRECT cold: reads the n_cold activated experts from Grace via a compact
    # cold tier. The pool tensors are sliced to the first n_cold experts.
    dcq13 = cq13[:n_cold] if n_cold else cq13[:0]
    dcs13 = cs13[:n_cold] if n_cold else cs13[:0]
    dcq2 = cq2[:n_cold] if n_cold else cq2[:0]
    dcs2 = cs2[:n_cold] if n_cold else cs2[:0]
    if n_cold:
        cold_direct, _ = build_chain(a13, a2, dcq13, dcs13, dcq2, dcs2, cmap, topo, ws(), ws(), m, device)
    # STAGED cold: gather the n_cold activated experts' slices into HBM slots.
    sq13 = torch.empty_like(dcq13); ss13 = torch.empty_like(dcs13)
    sq2 = torch.empty_like(dcq2); ss2 = torch.empty_like(dcs2)

    def transfer():
        for j in range(n_cold):
            sq13[j].copy_(cq13[j], non_blocking=True)
            ss13[j].copy_(cs13[j], non_blocking=True)
            sq2[j].copy_(cq2[j], non_blocking=True)
            ss2[j].copy_(cs2[j], non_blocking=True)
    nbytes = sum(t.numel() * t.element_size() for t in (sq13, ss13, sq2, ss2))
    if n_cold:
        cold_staged, _ = build_chain(a13, a2, sq13, ss13, sq2, ss2, cmap, topo, ws(), ws(), m, device)

    main = torch.cuda.current_stream()
    aux = torch.cuda.Stream()

    # --- solo references (with L2 flush) ---
    def solo(fn):
        for _ in range(6):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            l2.flush(); torch.cuda.synchronize()
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(main); fn(); e.record(main); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000.0)
        ts.sort(); return ts[len(ts) // 2]

    t_hot = solo(hot_chain)
    t_cold_grace = solo(cold_direct) if n_cold else 0.0
    transfer(); torch.cuda.synchronize()
    t_cold_hbm = solo(cold_staged) if n_cold else 0.0
    t_transfer = solo(transfer) if n_cold else 0.0
    c2c_gbs = nbytes / (t_transfer * 1e-6) / 1e9 if (n_cold and t_transfer > 0) else 0.0

    # --- concurrent transfer+hot (does the copy dilate hot?) ---
    if n_cold:
        def th(start_event=None):
            with torch.cuda.stream(aux):
                transfer()
            hot_chain()
        for _ in range(6):
            th()
        torch.cuda.synchronize()
        hts = []
        for _ in range(iters):
            l2.flush(); torch.cuda.synchronize()
            s = torch.cuda.Event(True); he = torch.cuda.Event(True)
            s.record(main)
            with torch.cuda.stream(aux):
                transfer()
            hot_chain(); he.record(main)
            torch.cuda.synchronize()
            hts.append(s.elapsed_time(he) * 1000.0)
        hts.sort(); t_hot_under_transfer = hts[len(hts) // 2]
        hot_dil = t_hot_under_transfer / t_hot - 1
    else:
        t_hot_under_transfer = t_hot
        hot_dil = 0.0

    # --- DIRECT control: hot + cold-from-Grace, production fork/join, common start ---
    def direct(start_event=None):
        if start_event is not None:
            aux.wait_event(start_event)
        if n_cold:
            with torch.cuda.stream(aux):
                cold_direct()
        hot_chain()
        if n_cold:
            main.wait_stream(aux)
    t_direct = time_common_start(direct, main, l2, warmup=8, iters=iters)

    # --- STAGED: {transfer || hot} then cold-from-HBM, common start + final join ---
    def staged(start_event=None):
        if start_event is not None:
            aux.wait_event(start_event)
        if n_cold:
            with torch.cuda.stream(aux):
                transfer()
            main.wait_stream(aux)   # cold needs the staged weights
            cold_staged()
        hot_chain()
    t_staged = time_common_start(staged, main, l2, warmup=8, iters=iters)

    est = (max(t_hot_under_transfer, t_transfer) + t_cold_hbm) if n_cold else t_hot
    frac = t_staged / t_direct - 1
    print(f"  m={m:3d} hot={n_hot} cold={n_cold} | hot={t_hot:.1f} coldG={t_cold_grace:.1f} "
          f"coldH={t_cold_hbm:.1f} xfer={t_transfer:.1f}({c2c_gbs:.0f}GB/s) dil={hot_dil:+.0%} | "
          f"direct={t_direct:.1f} staged={t_staged:.1f} est={est:.1f} | staged vs direct {frac:+.1%} "
          f"{'HELP' if frac < -0.01 else ('HURT' if frac > 0.01 else 'flat')}")
    return {
        "m": m, "n_hot": n_hot, "n_cold": n_cold, "transfer_mb": nbytes / 1e6,
        "hot_us": t_hot, "cold_grace_us": t_cold_grace, "cold_hbm_us": t_cold_hbm,
        "transfer_us": t_transfer, "c2c_gbs": c2c_gbs, "hot_dil_under_transfer": hot_dil,
        "direct_us": t_direct, "staged_us": t_staged, "est_us": est, "staged_vs_direct": frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numa-node", type=int, default=-1)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="results-fullchain.json")
    ap.add_argument("--selftest", action="store_true", help="one cell, verbose")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    device = torch.device("cuda:0")
    if args.numa_node < 0:
        args.numa_node = detect_grace_numa_node(0)
    print(f"=== Full-chain staging probe (q4 regime)  Grace NUMA {args.numa_node} ===")
    l2 = L2Flusher(device)

    # Grid centered on the measured c1q4 point (m=4, ~5 hot, ~1 cold, 18.5% cold),
    # plus neighbors and a c4q4 bridge (m=16). cold=0 is the no-staging control.
    if args.selftest:
        grid = [(4, 6, 1)]
    else:
        grid = []
        for n_hot in (4, 6, 8):
            for n_cold in (0, 1, 2, 4):
                grid.append((4, n_hot, n_cold))
        grid += [(16, 6, 1), (16, 6, 4), (16, 15, 4)]  # c4q4 bridge + straggler
    out = []
    for (m, n_hot, n_cold) in grid:
        # release the previous cell's tier allocations so fragmentation/thermal
        # does not shift whole cells (observed as bimodal hot times).
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            out.append(run_cell(m, n_hot, n_cold, device, args.numa_node, l2, args.iters))
        except Exception:
            traceback.print_exc()
            out.append({"m": m, "n_hot": n_hot, "n_cold": n_cold, "error": traceback.format_exc()})
    json.dump({"args": vars(args), "results": out}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    helps = sum(1 for r in out if "error" not in r and r["staged_vs_direct"] < -0.01)
    hurts = sum(1 for r in out if "error" not in r and r["staged_vs_direct"] > 0.01)
    print(f"cells: {len(out)}  help(>1%): {helps}  hurt(>1%): {hurts}  flat: {len(out)-helps-hurts}")


if __name__ == "__main__":
    sys.exit(main())
