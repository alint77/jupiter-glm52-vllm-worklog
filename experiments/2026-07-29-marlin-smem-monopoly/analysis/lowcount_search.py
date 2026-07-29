"""Why low activated-expert counts do not benefit, and what grid does.

Two candidate causes at low counts:

1. The fixed 2/1 CTAs-per-SM policy is the wrong *shape* for a tiny tier. With
   one activated cold expert the cold tier has only `n_tiles` MN-tiles (32 for
   w13), so a 132-CTA launch makes Marlin split K four ways and cooperate
   through global barrier spin-locks plus an fp32 C_tmp reduction. More CTAs
   buys C2C bandwidth but pays barrier traffic.
2. The tiers still cannot co-reside, in which case no grid helps.

Everything is timed under CUDA graph replay, because the eager fork/join costs
~110 us per iteration on Booster - larger than the whole kernel at these sizes.
"""

import argparse
import importlib.util
import json
import os
import statistics

import torch

import vllm._custom_ops as ops

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
spec = importlib.util.spec_from_file_location(
    "hb", os.path.join(_REPO, "benchmarks", "kernels",
                       "benchmark_moe_wna16_marlin_decode.py"))
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)

from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_workspace_new,
)
from vllm.scalar_type import scalar_types  # noqa: E402

HIDDEN, INTER, TOP_K, BLOCK_M = 6144, 2048, 8, 16
QUANT = scalar_types.uint4b8
LEGACY, TIGHT = ops.MARLIN_SMEM_LEGACY, ops.MARLIN_SMEM_TIGHT


class Tier:
    def __init__(self, ids, k, n, dev, pinned, numa, ne, m, topk):
        self.q, self.s = hb.make_tier(len(ids), k, n, dev, pinned, numa)
        self.map = hb.tier_map(ids, ne, dev)
        self.ws = marlin_make_workspace_new(dev, 4)
        self.k, self.n = k, n
        self.tok, self.eids, self.npost, self.blocks = hb.align(
            topk, BLOCK_M, ne, self.map)
        self.c = torch.zeros((m * TOP_K, n), dtype=torch.bfloat16, device=dev)

    def call(self, a, w, m, smem, grid):
        ops.moe_wna16_marlin_gemm(
            a, self.c, self.q, None, self.s, None, None, None, None, None,
            self.ws, self.tok, self.eids, self.npost, w,
            moe_block_size=BLOCK_M, top_k=TOP_K, mul_topk_weights=False,
            b_q_type=QUANT, size_m=m, size_n=self.n, size_k=self.k,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
            is_zp_float=False, smem_mode=smem, grid_blocks=grid)


def capture(fn):
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


def time_graph(g, iters=60, rounds=5):
    out = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        b, e = torch.cuda.Event(True), torch.cuda.Event(True)
        b.record()
        for _ in range(iters):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        out.append(b.elapsed_time(e) * 1e3 / iters)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--hot", type=int, default=7)
    ap.add_argument("--cold", type=int, default=1)
    ap.add_argument("--numa-node", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ne = args.hot + args.cold
    topk = torch.tensor(
        [[(i * TOP_K + j) % ne for j in range(TOP_K)] for i in range(args.m)],
        dtype=torch.int32, device=dev)

    result = {"m": args.m, "hot": args.hot, "cold": args.cold, "layers": {}}
    for lname, (k, n) in (("w13", (HIDDEN, 2 * INTER)), ("w2", (INTER, HIDDEN))):
        hot = Tier(list(range(args.hot)), k, n, dev, False, args.numa_node, ne,
                   args.m, topk)
        cold = Tier(list(range(args.hot, ne)), k, n, dev, True, args.numa_node,
                    ne, args.m, topk)
        a = torch.randn((args.m, k), dtype=torch.bfloat16, device=dev) / 8
        w = torch.ones((args.m, TOP_K), dtype=torch.float32, device=dev)
        aux = torch.cuda.Stream()
        n_tiles = n // 128  # the auto tile is thread_n=128 at these shapes
        print(f"\n=== {lname}: k={k} n={n} | hot {args.hot} experts "
              f"({hot.blocks} blocks, ~{args.hot * n_tiles} mn-tiles) | "
              f"cold {args.cold} ({cold.blocks} blocks, "
              f"~{args.cold * n_tiles} mn-tiles) ===")

        def both(sh, gh, sc, gc):
            main = torch.cuda.current_stream()
            aux.wait_stream(main)
            with torch.cuda.stream(aux):
                cold.call(a, w, args.m, sc, gc)
            hot.call(a, w, args.m, sh, gh)
            main.wait_stream(aux)

        h_solo = time_graph(capture(lambda: hot.call(a, w, args.m, LEGACY, -1)))
        c_solo = time_graph(capture(lambda: cold.call(a, w, args.m, LEGACY, -1)))
        prod = time_graph(capture(lambda: both(LEGACY, -1, LEGACY, -1)))
        print(f"  hot solo {h_solo:7.1f} | cold solo {c_solo:7.1f} | "
              f"serial {h_solo + c_solo:7.1f} | ideal {max(h_solo, c_solo):7.1f} "
              f"| production union {prod:7.1f}")

        # what does the cold tier alone cost at each grid? isolates the
        # split-K/barrier tradeoff from any co-residency question
        print("  cold solo vs grid:", end=" ")
        cold_by_grid = {}
        for gc in (sms // 4, sms // 2, sms, sms * 2, sms * 3):
            t = time_graph(capture(lambda gc=gc: cold.call(a, w, args.m, TIGHT, gc)))
            cold_by_grid[gc] = t
            print(f"{gc}:{t:.1f}", end="  ")
        print()

        best = None
        rows = []
        for gh in (sms, sms * 2, sms * 3):
            line = []
            for gc in (sms // 4, sms // 2, sms, sms * 2):
                t = time_graph(capture(
                    lambda gh=gh, gc=gc: both(TIGHT, gh, TIGHT, gc)))
                rows.append({"hot_grid": gh, "cold_grid": gc, "union": t})
                line.append(f"{gc:>4}:{t:7.1f}")
                if best is None or t < best["union"]:
                    best = rows[-1]
            print(f"  hot grid {gh:>4} | " + "  ".join(line))
        print(f"  best {best['hot_grid']}/{best['cold_grid']} = "
              f"{best['union']:.1f} us vs production {prod:.1f} "
              f"({best['union'] / prod - 1:+.1%}), ideal {max(h_solo, c_solo):.1f}")
        result["layers"][lname] = {
            "hot_solo": h_solo, "cold_solo": c_solo, "prod": prod,
            "cold_by_grid": cold_by_grid, "grid_search": rows, "best": best,
        }

    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
