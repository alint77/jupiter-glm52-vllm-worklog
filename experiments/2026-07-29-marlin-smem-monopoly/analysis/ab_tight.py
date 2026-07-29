"""Interleaved A/B of the production launch vs the tight-smem launch.

The login GPU is shared, so absolute times drift. Each round measures every
variant back-to-back and we report per-round ratios; drift cancels.
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


class Tier:
    def __init__(self, ids, k, n, dev, pinned, numa, num_experts, m, topk):
        self.q, self.s = hb.make_tier(len(ids), k, n, dev, pinned, numa)
        self.map = hb.tier_map(ids, num_experts, dev)
        self.ws = marlin_make_workspace_new(dev, 4)
        self.k, self.n = k, n
        self.tok, self.eids, self.npost, self.blocks = hb.align(
            topk, BLOCK_M, num_experts, self.map
        )
        self.c = torch.zeros((m * TOP_K, n), dtype=torch.bfloat16, device=dev)
        self.c_tmp = torch.empty(
            min(n * self.tok.shape[0], 132 * 4 * BLOCK_M * 256),
            dtype=torch.float32, device=dev,
        )

    def call(self, a, w, m, smem, grid):
        marlin_tight.gemm(a, self.c, self.q, self.s, self.c_tmp, self.ws,
                          self.tok, self.eids, self.npost, w, BLOCK_M, TOP_K,
                          False, m, self.n, self.k, -1, -1, -1, smem, grid)


def timeit(fn, iters=100):
    torch.cuda.synchronize()
    beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
    beg.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return beg.elapsed_time(end) * 1e3 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--hot", type=int, default=19)
    ap.add_argument("--cold", type=int, default=3)
    ap.add_argument("--cold-share", type=float, default=0.13)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--numa-node", type=int, default=0)
    ap.add_argument("--out", default=None)
    a_ = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    ne = a_.hot + a_.cold
    hot_ids, cold_ids = list(range(a_.hot)), list(range(a_.hot, ne))
    topk = hb.global_routing(a_.m, hot_ids, cold_ids, ne, dev, 0, a_.cold_share)

    # w13 (n = 2*intermediate) and w2 (n = hidden, k = intermediate)
    layers = {
        "w13": (HIDDEN, 2 * INTER),
        "w2": (INTER, HIDDEN),
    }
    results = {}
    for lname, (k, n) in layers.items():
        hot = Tier(hot_ids, k, n, dev, False, a_.numa_node, ne, a_.m, topk)
        cold = Tier(cold_ids, k, n, dev, True, a_.numa_node, ne, a_.m, topk)
        a = torch.randn((a_.m, k), dtype=torch.bfloat16, device=dev) / 8
        w = torch.ones((a_.m, TOP_K), dtype=torch.float32, device=dev)
        aux = torch.cuda.Stream()

        def both(sh, gh, sc, gc):
            main = torch.cuda.current_stream()
            aux.wait_stream(main)
            with torch.cuda.stream(aux):
                cold.call(a, w, a_.m, sc, gc)
            hot.call(a, w, a_.m, sh, gh)
            main.wait_stream(aux)

        variants = {
            "production": lambda: both(LEGACY, -1, LEGACY, -1),
            "tight_264_132": lambda: both(TIGHT, 264, TIGHT, 132),
            "tight_264_66": lambda: both(TIGHT, 264, TIGHT, 66),
            "tight_396_132": lambda: both(TIGHT, 396, TIGHT, 132),
            "hot_solo_legacy": lambda: hot.call(a, w, a_.m, LEGACY, -1),
            "hot_solo_tight396": lambda: hot.call(a, w, a_.m, TIGHT, 396),
            "cold_solo_legacy": lambda: cold.call(a, w, a_.m, LEGACY, -1),
        }
        for f in variants.values():
            for _ in range(20):
                f()
        torch.cuda.synchronize()

        samples = {kk: [] for kk in variants}
        for _ in range(a_.rounds):
            for kk, f in variants.items():
                samples[kk].append(timeit(f))
        med = {kk: statistics.median(v) for kk, v in samples.items()}
        # per-round ratio, so GPU drift cancels
        ratio = statistics.median(
            [t / p for t, p in zip(samples["tight_264_132"],
                                   samples["production"])]
        )
        results[lname] = dict(median_us=med, spread={
            kk: (max(v) - min(v)) / statistics.median(v) for kk, v in samples.items()
        }, tight_vs_prod_ratio=ratio)

        print(f"\n=== {lname}  (k={k}, n={n}, m={a_.m}, "
              f"hot={a_.hot}/cold={a_.cold}) ===")
        for kk in variants:
            print(f"  {kk:<20} {med[kk]:8.1f} us   "
                  f"(spread {results[lname]['spread'][kk]:.1%})")
        ideal = max(med["hot_solo_tight396"], med["cold_solo_legacy"])
        print(f"  ideal max(hot,cold)  {ideal:8.1f} us")
        print(f"  --> tight_264_132 vs production: {ratio - 1:+.1%} "
              f"(per-round median ratio)")

    if a_.out:
        json.dump(dict(args=vars(a_), results=results), open(a_.out, "w"), indent=2)


if __name__ == "__main__":
    main()
