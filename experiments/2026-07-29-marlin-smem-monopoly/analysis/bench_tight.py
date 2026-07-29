"""Does the tight-shared-memory fix let hot and cold Marlin actually overlap?

Legacy  = upstream launch (dynamic smem = deviceSharedMemOptin / blocks_per_sm)
Tight   = launch with sh_cache_size, caller-chosen grid

Same kernel, same tile, same routing; only the launch geometry differs.
"""

import argparse
import importlib.util
import json
import os
import sys

import torch

sys.path.insert(0, "/e/project1/profound/alint77/vllm/benchmarks/kernels")
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

HIDDEN, INTER, GROUP, TOP_K, BLOCK_M = 6144, 2048, 128, 8, 16
SMS = 132
LEGACY, TIGHT = -1, -2


def numa_node_for(dev: int) -> int:
    try:
        from vllm.model_executor.offloader.grace import get_device_numa_node

        return get_device_numa_node(dev)
    except Exception as e:  # login node fallback
        print(f"  (numa autodetect failed: {e}; using node 0)")
        return 0


class Tier:
    def __init__(self, ids, k, n, dev, pinned, numa, num_experts):
        self.ids = ids
        self.q, self.s = hb.make_tier(len(ids), k, n, dev, pinned, numa)
        self.map = hb.tier_map(ids, num_experts, dev)
        self.ws = marlin_make_workspace_new(dev, 4)
        self.k, self.n = k, n

    def align(self, topk_ids, num_experts):
        self.tok, self.eids, self.npost, self.blocks = hb.align(
            topk_ids, BLOCK_M, num_experts, self.map
        )

    def alloc(self, m, dev):
        self.c = torch.empty((m * TOP_K, self.n), dtype=torch.bfloat16, device=dev)
        n_tmp = min(self.n * self.tok.shape[0], SMS * 4 * BLOCK_M * 256)
        self.c_tmp = torch.empty(n_tmp, dtype=torch.float32, device=dev)

    def call(self, a, weights, m, smem, grid, tk=-1, tn=-1, bps=-1):
        marlin_tight.gemm(
            a, self.c, self.q, self.s, self.c_tmp, self.ws, self.tok, self.eids,
            self.npost, weights, BLOCK_M, TOP_K, False, m, self.n, self.k,
            tk, tn, bps, smem, grid,
        )


def time_union(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
    beg.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return beg.elapsed_time(end) * 1e3 / iters  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--hot", type=int, default=19)
    ap.add_argument("--cold", type=int, default=3)
    ap.add_argument("--cold-share", type=float, default=0.13)
    ap.add_argument("--numa-node", type=int, default=-1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    numa = args.numa_node if args.numa_node >= 0 else numa_node_for(0)
    n_experts = args.hot + args.cold
    hot_ids = list(range(args.hot))
    cold_ids = list(range(args.hot, n_experts))
    n13 = 2 * INTER

    print(f"m={args.m} hot={args.hot} (HBM) cold={args.cold} (Grace numa {numa}) "
          f"w13 k={HIDDEN} n={n13}")
    hot = Tier(hot_ids, HIDDEN, n13, dev, False, numa, n_experts)
    cold = Tier(cold_ids, HIDDEN, n13, dev, True, numa, n_experts)

    topk = hb.global_routing(args.m, hot_ids, cold_ids, n_experts, dev, 0,
                             args.cold_share)
    for t in (hot, cold):
        t.align(topk, n_experts)
        t.alloc(args.m, dev)
    print(f"  padded token blocks: hot={hot.blocks} cold={cold.blocks}")

    a = torch.randn((args.m, HIDDEN), dtype=torch.bfloat16, device=dev) / 8
    weights = torch.ones((args.m, TOP_K), dtype=torch.float32, device=dev)

    # ---- correctness: tight must be bit-identical to legacy ----------------
    os.environ["MARLIN_TIGHT_VERBOSE"] = "1"

    def run_once(tier, smem, grid):
        # rows of tokens routed to the *other* tier are never written, so the
        # buffer must be zeroed for the comparison to be meaningful
        tier.c.zero_()
        tier.call(a, weights, args.m, smem, grid)
        torch.cuda.synchronize()
        return tier.c.clone()

    ref_hot = run_once(hot, LEGACY, -1)
    ref_cold = run_once(cold, LEGACY, -1)
    del os.environ["MARLIN_TIGHT_VERBOSE"]

    checks = {}
    for name, tier, ref in (("hot", hot, ref_hot), ("cold", cold, ref_cold)):
        rerun = run_once(tier, LEGACY, -1)
        checks[f"{name}_legacy_repeatable"] = bool(torch.equal(rerun, ref))
        for grid in (132, 264, 396, 528):
            got = run_once(tier, TIGHT, grid)
            same = torch.equal(got, ref)
            dmax = (got.float() - ref.float()).abs().max().item()
            checks[f"{name}_tight_g{grid}"] = dict(bitexact=bool(same), max_abs=dmax)
            print(f"  {name}: legacy-repeatable="
                  f"{checks[f'{name}_legacy_repeatable']} | tight grid={grid} "
                  f"bit-exact={same} max|diff|={dmax:.3g}")

    aux = torch.cuda.Stream()

    def both(smem_h, grid_h, smem_c, grid_c):
        main = torch.cuda.current_stream()
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            cold.call(a, weights, args.m, smem_c, grid_c)
        hot.call(a, weights, args.m, smem_h, grid_h)
        main.wait_stream(aux)

    solo_hot = time_union(lambda: hot.call(a, weights, args.m, LEGACY, -1))
    solo_cold = time_union(lambda: cold.call(a, weights, args.m, LEGACY, -1))
    prod = time_union(lambda: both(LEGACY, -1, LEGACY, -1))
    print(f"\nlegacy: hot solo {solo_hot:.1f} us | cold solo {solo_cold:.1f} us "
          f"| serial {solo_hot + solo_cold:.1f} | union {prod:.1f} us "
          f"({prod / max(solo_hot, solo_cold) - 1:+.0%} vs ideal)")

    rows = []
    print(f"\n{'hot grid':>9} {'cold grid':>10} {'hot solo':>9} {'cold solo':>10} "
          f"{'union':>8} {'vs prod':>8} {'vs ideal':>9}")
    for gh in (264, 396, 528):
        for gc in (66, 132, 198):
            sh = time_union(lambda: hot.call(a, weights, args.m, TIGHT, gh))
            sc = time_union(lambda: cold.call(a, weights, args.m, TIGHT, gc))
            u = time_union(lambda: both(TIGHT, gh, TIGHT, gc))
            rows.append(dict(hot_grid=gh, cold_grid=gc, hot_solo=sh, cold_solo=sc,
                             union=u, vs_prod=u / prod - 1,
                             vs_ideal=u / max(sh, sc) - 1))
            print(f"{gh:>9} {gc:>10} {sh:>9.1f} {sc:>10.1f} {u:>8.1f} "
                  f"{u / prod - 1:>+7.1%} {u / max(sh, sc) - 1:>+8.1%}")

    best = min(rows, key=lambda r: r["union"])
    print(f"\nbest: hot grid {best['hot_grid']}, cold grid {best['cold_grid']} -> "
          f"{best['union']:.1f} us vs production {prod:.1f} us "
          f"({best['vs_prod']:+.1%})")

    if args.out:
        json.dump(dict(args=vars(args), checks=checks, solo_hot=solo_hot,
                       solo_cold=solo_cold, production_union=prod, sweep=rows),
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
