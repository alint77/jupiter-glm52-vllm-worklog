"""Two-tier Marlin union, before and after the tight-smem fix, over the
activated-expert counts a GH200 EP4 rank actually sees.

q4..q16 verify tokens -> 32..128 routed assignments per rank -> roughly 8..32
distinct activated experts per rank per layer, ~20% of them cold.

Routing is built so that exactly `hot + cold` experts are activated, each token
getting `top_k` distinct experts, which is what sets Marlin's padded block count.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys

import torch

spec = importlib.util.spec_from_file_location(
    "hb", "/e/project1/profound/alint77/vllm/benchmarks/kernels/"
    "benchmark_moe_wna16_marlin_decode.py"
)
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "marlin_tight", "build"))
import marlin_tight  # noqa: E402

from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_workspace_new,
)

HIDDEN, INTER, TOP_K, BLOCK_M = 6144, 2048, 8, 16
LEGACY, TIGHT = -1, -2
SMS = 132


def routing(m, n_experts, dev):
    """Every one of `n_experts` is activated; each token gets top_k distinct."""
    rows = []
    for i in range(m):
        rows.append([(i * TOP_K + j) % n_experts for j in range(TOP_K)])
    return torch.tensor(rows, dtype=torch.int32, device=dev)


class Tier:
    def __init__(self, ids, k, n, dev, pinned, numa, n_experts, m, topk, wcache):
        key = (len(ids), k, n, pinned)
        if key not in wcache:
            wcache[key] = hb.make_tier(len(ids), k, n, dev, pinned, numa)
        self.q, self.s = wcache[key]
        self.map = hb.tier_map(ids, n_experts, dev)
        self.ws = marlin_make_workspace_new(dev, 4)
        self.k, self.n = k, n
        self.tok, self.eids, self.npost, self.blocks = hb.align(
            topk, BLOCK_M, n_experts, self.map
        )
        self.c = torch.zeros((m * TOP_K, n), dtype=torch.bfloat16, device=dev)
        self.c_tmp = torch.empty(
            min(n * self.tok.shape[0], SMS * 4 * BLOCK_M * 256),
            dtype=torch.float32, device=dev,
        )

    def call(self, a, w, m, smem, grid):
        marlin_tight.gemm(a, self.c, self.q, self.s, self.c_tmp, self.ws,
                          self.tok, self.eids, self.npost, w, BLOCK_M, TOP_K,
                          False, m, self.n, self.k, -1, -1, -1, smem, grid)


def timeit(fn, iters=50):
    torch.cuda.synchronize()
    b, e = torch.cuda.Event(True), torch.cuda.Event(True)
    b.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return b.elapsed_time(e) * 1e3 / iters


def cell(m, n_hot, n_cold, dev, numa, rounds, wcache):
    ne = n_hot + n_cold
    topk = routing(m, ne, dev)
    hot_ids, cold_ids = list(range(n_hot)), list(range(n_hot, ne))
    total_prod = total_tight = 0.0
    per_layer = {}
    for lname, (k, n) in (("w13", (HIDDEN, 2 * INTER)), ("w2", (INTER, HIDDEN))):
        hot = Tier(hot_ids, k, n, dev, False, numa, ne, m, topk, wcache)
        cold = Tier(cold_ids, k, n, dev, True, numa, ne, m, topk, wcache)
        a = torch.randn((m, k), dtype=torch.bfloat16, device=dev) / 8
        w = torch.ones((m, TOP_K), dtype=torch.float32, device=dev)
        aux = torch.cuda.Stream()

        def both(sh, gh, sc, gc):
            main = torch.cuda.current_stream()
            aux.wait_stream(main)
            with torch.cuda.stream(aux):
                cold.call(a, w, m, sc, gc)
            hot.call(a, w, m, sh, gh)
            main.wait_stream(aux)

        variants = {
            "prod": lambda: both(LEGACY, -1, LEGACY, -1),
            "tight": lambda: both(TIGHT, 264, TIGHT, 132),
            "hot_solo": lambda: hot.call(a, w, m, LEGACY, -1),
            "cold_solo": lambda: cold.call(a, w, m, LEGACY, -1),
        }
        for f in variants.values():
            for _ in range(15):
                f()
        s = {kk: [] for kk in variants}
        for _ in range(rounds):
            for kk, f in variants.items():
                s[kk].append(timeit(f))
        med = {kk: statistics.median(v) for kk, v in s.items()}
        ratio = statistics.median([t / p for t, p in zip(s["tight"], s["prod"])])
        per_layer[lname] = dict(med, ratio=ratio, hot_blocks=hot.blocks,
                                cold_blocks=cold.blocks)
        total_prod += med["prod"]
        total_tight += med["tight"]
    return per_layer, total_prod, total_tight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--numa-node", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)

    # (verify tokens m, activated experts on this rank, cold experts)
    grid = [
        (4, 8, 1), (4, 8, 2), (4, 12, 2), (4, 12, 4),
        (8, 12, 2), (8, 16, 3), (8, 16, 5), (8, 24, 5),
        (12, 16, 3), (12, 24, 5), (12, 24, 7),
        (16, 24, 5), (16, 32, 6), (16, 32, 10),
    ]
    wcache: dict = {}
    rows = []
    print(f"{'m':>3} {'act':>4} {'hot':>4} {'cold':>5} {'cold%':>6} "
          f"{'prod us':>9} {'tight us':>9} {'delta':>8} "
          f"{'w13':>8} {'w2':>8}")
    for m, act, n_cold in grid:
        n_hot = act - n_cold
        per_layer, tp, tt = cell(m, n_hot, n_cold, dev, args.numa_node,
                                 args.rounds, wcache)
        rows.append(dict(m=m, activated=act, hot=n_hot, cold=n_cold,
                         prod_us=tp, tight_us=tt, delta=tt / tp - 1,
                         layers=per_layer))
        print(f"{m:>3} {act:>4} {n_hot:>4} {n_cold:>5} {n_cold / act:>5.0%} "
              f"{tp:>9.1f} {tt:>9.1f} {tt / tp - 1:>+7.1%} "
              f"{per_layer['w13']['ratio'] - 1:>+7.1%} "
              f"{per_layer['w2']['ratio'] - 1:>+7.1%}")

    d = [r["delta"] for r in rows]
    print(f"\nfull w13+w2 chain, {len(rows)} cells: median {statistics.median(d):+.1%}, "
          f"best {min(d):+.1%}, worst {max(d):+.1%}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
