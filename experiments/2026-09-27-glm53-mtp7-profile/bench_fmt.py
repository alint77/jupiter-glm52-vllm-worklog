#!/usr/bin/env python3
"""The one-kernel tiered decode MoE, MXFP4 (MiMo) against INT4 (GLM), same shapes.

Both models are hidden 6144 / expert 2048 / top-8 over an 8-token verify, so one
layer call differs only in the weight format. Tiers are built with the kernel
test's own converters (tests/kernels/moe/test_tiered_decode_moe.py); routing and
timing are ../2026-09-27-dak-tiered-moe-kernel/bench_vllm_decode.py's (MiMo's
token mix, graph replay of 20 calls, best of 10).

    bench_fmt.py --fmt int4 --grid 9,0 0,2 9,2 ...   (JSON lines on stdout)

Run NUMA-bound on the GPU's Grace node.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = load("tiered_test", REPO / "tests/kernels/moe/test_tiered_decode_moe.py")
B = load("dak_bench", HERE.parent / "2026-09-27-dak-tiered-moe-kernel/bench_vllm_decode.py")


def tier(fmt: str, count: int, device, cold: bool, numa: int, gen,
         size: int = 0) -> tuple[dict, list]:
    """count real experts; with size > count, an uninitialised pool of size
    experts of the same layout (speed does not depend on values)."""
    from vllm.model_executor.offloader.grace import GraceAllocation

    experts, _, marlin, _ = T.FORMATS[fmt]
    t = marlin(*experts(count, gen), device)
    if size > count:
        shapes = {k: (size, *v.shape[1:]) for k, v in t.items()}
        if not cold:
            t = {k: torch.empty(shapes[k], dtype=v.dtype, device=device) for k, v in t.items()}
        else:
            out, keep = {}, []
            for k, v in t.items():
                a = GraceAllocation.allocate_pinned(shapes[k], v.dtype, device.index or 0, numa)
                keep.append(a)
                out[k] = a.cuda_alias
            return out, keep
    if not cold:
        return t, []
    out, keep = {}, []
    for name, x in t.items():
        a = GraceAllocation.allocate_pinned(tuple(x.shape), x.dtype, device.index or 0, numa)
        a.copy_from(x.cpu())
        keep.append(a)
        out[name] = a.cuda_alias
    del t
    torch.cuda.empty_cache()
    return out, keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", choices=sorted(T.FORMATS), required=True)
    ap.add_argument("--grid", nargs="+", required=True, help="hot,cold cells")
    ap.add_argument("--numa-node", type=int, default=0)
    ap.add_argument("--pool-hot", type=int, default=0,
                    help="production-sized hot pool, experts picked at random per call")
    ap.add_argument("--pool-cold", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(0)
    cells = [tuple(map(int, c.split(","))) for c in args.grid]
    pool_hot = max(16, max(h for h, _ in cells), args.pool_hot)
    pool_cold = max(8, max(c for _, c in cells), args.pool_cold)
    hot, _ = tier(args.fmt, 16, device, False, args.numa_node, gen, pool_hot)
    cold, keep = tier(args.fmt, 8, device, True, args.numa_node, gen, pool_cold)
    x = (torch.randn((B.TOKENS, B.HIDDEN), generator=gen) * 0.3).to(torch.bfloat16).to(device)
    for h, c in cells:
        calls = [B.routing(h, c, pool_hot, pool_cold, r, gen, device) for r in range(20)]
        if args.pool_hot or args.pool_cold:  # fresh experts every call
            for ids, w, hmap, cmap in calls:
                on = hmap >= 0
                hmap[on] = torch.randint(0, pool_hot, (int(on.sum()),), device=device,
                                         dtype=hmap.dtype)
                on = cmap >= 0
                cmap[on] = torch.randint(0, pool_cold, (int(on.sum()),), device=device,
                                         dtype=cmap.dtype)
        print(json.dumps({"fmt": args.fmt, "hot": h, "cold": c,
                          "us": round(B.wall_us(x, calls, hot, cold), 1)}), flush=True)
    del keep


if __name__ == "__main__":
    sys.exit(main())
