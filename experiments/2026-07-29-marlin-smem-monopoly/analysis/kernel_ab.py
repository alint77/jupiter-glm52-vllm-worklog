"""Booster kernel-level A/B of the tight-shared-memory Marlin launch.

Uses the in-tree `ops.moe_wna16_marlin_gemm`, so this also validates the shipped
integration rather than a staged copy. Must run on a Booster node: the login
GH200 has a different power cap (and therefore different clocks) and a slightly
different C2C, so neither the absolute times nor the hot/cold balance transfer.

Every variant is measured inside each round and reported as a per-round paired
ratio, so any drift or co-tenancy cancels.
"""

import argparse
import gc
import importlib.util
import json
import os
import statistics
import sys

import torch

import vllm._custom_ops as ops

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
spec = importlib.util.spec_from_file_location(
    "hb",
    os.path.join(_REPO, "benchmarks", "kernels",
                 "benchmark_moe_wna16_marlin_decode.py"),
)
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)

from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_workspace_new,
)
from vllm.scalar_type import scalar_types  # noqa: E402

HIDDEN, INTER, TOP_K, BLOCK_M = 6144, 2048, 8, 16
QUANT = scalar_types.uint4b8
LEGACY = ops.MARLIN_SMEM_LEGACY
TIGHT = ops.MARLIN_SMEM_TIGHT


def routing(m, n_experts, dev):
    """Activate exactly `n_experts`, `top_k` distinct experts per token."""
    return torch.tensor(
        [[(i * TOP_K + j) % n_experts for j in range(TOP_K)] for i in range(m)],
        dtype=torch.int32, device=dev,
    )


class Tier:
    def __init__(self, ids, k, n, dev, pinned, numa, n_experts, m, topk, cache):
        key = (len(ids), k, n, pinned)
        if key not in cache:
            cache[key] = hb.make_tier(len(ids), k, n, dev, pinned, numa)
        self.q, self.s = cache[key]
        self.map = hb.tier_map(ids, n_experts, dev)
        self.ws = marlin_make_workspace_new(dev, 4)
        self.k, self.n = k, n
        self.tok, self.eids, self.npost, self.blocks = hb.align(
            topk, BLOCK_M, n_experts, self.map
        )
        self.c = torch.zeros((m * TOP_K, n), dtype=torch.bfloat16, device=dev)
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count

    def call(self, a, w, m, smem, grid):
        ops.moe_wna16_marlin_gemm(
            a, self.c, self.q, None, self.s, None, None, None, None, None,
            self.ws, self.tok, self.eids, self.npost, w,
            moe_block_size=BLOCK_M, top_k=TOP_K, mul_topk_weights=False,
            b_q_type=QUANT, size_m=m, size_n=self.n, size_k=self.k,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
            is_zp_float=False, smem_mode=smem, grid_blocks=grid,
        )


def timeit(fn, iters=50):
    torch.cuda.synchronize()
    b, e = torch.cuda.Event(True), torch.cuda.Event(True)
    b.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return b.elapsed_time(e) * 1e3 / iters


def capture(fn):
    """Capture one call into a CUDA graph, as the production runner does.

    Timing the two-tier fork/join eagerly charges two stream barriers per
    iteration, which stops the host running ahead and exposes launch latency on
    every iteration. That cost is a harness artifact: production replays the
    fork/join from inside a graph, where it is a dependency edge with no host
    round-trip. On Booster the eager floor is ~110 us, which is larger than the
    entire kernel at low activated-expert counts and hides any difference there.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def time_graph(g, iters=50):
    torch.cuda.synchronize()
    b, e = torch.cuda.Event(True), torch.cuda.Event(True)
    b.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return b.elapsed_time(e) * 1e3 / iters


def cell(m, n_hot, n_cold, dev, numa, rounds, cache, check=False):
    # `cache` is per-cell: pinned Grace tiers must be released between cells or
    # a 120 GB Grace node runs out and later allocations spill off-node.
    ne = n_hot + n_cold
    topk = routing(m, ne, dev)
    hot_ids, cold_ids = list(range(n_hot)), list(range(n_hot, ne))
    out = {"m": m, "hot": n_hot, "cold": n_cold, "layers": {}}
    tp = tt = 0.0
    for lname, (k, n) in (("w13", (HIDDEN, 2 * INTER)), ("w2", (INTER, HIDDEN))):
        hot = Tier(hot_ids, k, n, dev, False, numa, ne, m, topk, cache)
        cold = Tier(cold_ids, k, n, dev, True, numa, ne, m, topk, cache)
        a = torch.randn((m, k), dtype=torch.bfloat16, device=dev) / 8
        w = torch.ones((m, TOP_K), dtype=torch.float32, device=dev)
        aux = torch.cuda.Stream()
        # shipped policy: hot 2 CTAs/SM, cold 1
        g_hot, g_cold = hot.sms * 2, hot.sms

        if check:
            for tier, grid in ((hot, hot.sms * 3), (cold, hot.sms * 3)):
                tier.c.zero_()
                tier.call(a, w, m, LEGACY, -1)
                torch.cuda.synchronize()
                ref = tier.c.clone()
                tier.c.zero_()
                tier.call(a, w, m, TIGHT, grid)
                torch.cuda.synchronize()
                out.setdefault("checks", {})[f"{lname}_same_grid_bitexact"] = bool(
                    torch.equal(tier.c, ref)
                )
                tier.c.zero_()
                tier.call(a, w, m, TIGHT, g_hot if tier is hot else g_cold)
                torch.cuda.synchronize()
                out["checks"][f"{lname}_shipped_grid_max_abs"] = float(
                    (tier.c.float() - ref.float()).abs().max()
                )

        def both(sh, ghh, sc, gcc):
            main = torch.cuda.current_stream()
            aux.wait_stream(main)
            with torch.cuda.stream(aux):
                cold.call(a, w, m, sc, gcc)
            hot.call(a, w, m, sh, ghh)
            main.wait_stream(aux)

        variants = {
            "prod": lambda: both(LEGACY, -1, LEGACY, -1),
            "tight": lambda: both(TIGHT, g_hot, TIGHT, g_cold),
            "hot_solo": lambda: hot.call(a, w, m, LEGACY, -1),
            "cold_solo": lambda: cold.call(a, w, m, LEGACY, -1),
        }
        for f in variants.values():
            for _ in range(15):
                f()
        graphs = {kk: capture(f) for kk, f in variants.items()}
        s = {kk: [] for kk in variants}
        sg = {kk: [] for kk in variants}
        for _ in range(rounds):
            for kk, f in variants.items():
                s[kk].append(timeit(f))
                sg[kk].append(time_graph(graphs[kk]))
        med = {kk: statistics.median(v) for kk, v in s.items()}
        med.update({f"g_{kk}": statistics.median(v) for kk, v in sg.items()})
        med["ratio"] = statistics.median(
            [t / p for t, p in zip(s["tight"], s["prod"])]
        )
        med["g_ratio"] = statistics.median(
            [t / p for t, p in zip(sg["tight"], sg["prod"])]
        )
        med["hot_blocks"], med["cold_blocks"] = hot.blocks, cold.blocks
        out["layers"][lname] = med
        tp += med["g_prod"]
        tt += med["g_tight"]
    out["prod_us"], out["tight_us"], out["delta"] = tp, tt, tt / tp - 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--numa-node", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    smem_sm = props.shared_memory_per_multiprocessor
    optin = torch.cuda.get_device_properties(dev).shared_memory_per_block_optin
    print(f"{props.name}, {props.multi_processor_count} SMs, "
          f"numa node {args.numa_node}")
    print(f"shared memory: {smem_sm} B/SM, {optin} B/block optin -> "
          f"a bps=3 Marlin wave requests {3 * (optin // 3 - 1024)} B of "
          f"{smem_sm} B")

    grid = [
        (4, 8, 1), (4, 8, 2), (4, 12, 2), (4, 12, 4),
        (8, 12, 2), (8, 16, 3), (8, 16, 5), (8, 24, 5),
        (12, 16, 3), (12, 24, 5), (12, 24, 7),
        (16, 24, 5), (16, 32, 6), (16, 32, 10),
        (16, 22, 3),  # the c4 straggler shape used on the login node
    ]
    rows = []
    print(f"\n{'m':>3} {'act':>4} {'hot':>4} {'cold':>5} {'cold%':>6} "
          f"{'hot solo':>9} {'cold solo':>10} {'serial':>8} {'prod us':>9} "
          f"{'tight us':>9} {'ideal':>8} {'delta':>8} {'eager':>9}")
    for i, (m, act, n_cold) in enumerate(grid):
        r = cell(m, act - n_cold, n_cold, dev, args.numa_node, args.rounds,
                 {}, check=(i == 0))
        gc.collect()
        torch.cuda.empty_cache()
        rows.append(r)
        hs = sum(r["layers"][ln]["g_hot_solo"] for ln in ("w13", "w2"))
        cs = sum(r["layers"][ln]["g_cold_solo"] for ln in ("w13", "w2"))
        ideal = sum(max(r["layers"][ln]["g_hot_solo"],
                        r["layers"][ln]["g_cold_solo"])
                    for ln in ("w13", "w2"))
        eager_prod = sum(r["layers"][ln]["prod"] for ln in ("w13", "w2"))
        print(f"{m:>3} {act:>4} {act - n_cold:>4} {n_cold:>5} "
              f"{n_cold / act:>5.0%} {hs:>9.1f} {cs:>10.1f} {hs + cs:>8.1f} "
              f"{r['prod_us']:>9.1f} {r['tight_us']:>9.1f} {ideal:>8.1f} "
              f"{r['delta']:>+7.1%} {eager_prod:>9.1f}")

    d = [r["delta"] for r in rows]
    print(f"\nfull w13+w2 chain, {len(rows)} cells: median "
          f"{statistics.median(d):+.1%}, best {min(d):+.1%}, worst {max(d):+.1%}")
    if rows[0].get("checks"):
        print(f"correctness: {json.dumps(rows[0]['checks'])}")
    if args.out:
        json.dump(dict(device=props.name, sms=props.multi_processor_count,
                       rows=rows), open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
