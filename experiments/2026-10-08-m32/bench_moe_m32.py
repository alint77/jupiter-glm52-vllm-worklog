#!/usr/bin/env python3
"""One MoE layer call at M tokens, over the (hot, cold) cells that cover 95%
of rank-layer calls at that M (grid-m8-16-32.json, all ranks), INT4 tiers.

Per cell: 20 different routings (fresh distinct hot / cold slots from
production-sized pools; tokens per expert drawn from GLM's measured
distribution at this M), graph replay, best of 10 (bench_vllm_decode.wall_us
timing). Kernels: the tiered decode kernel in this tree ("decode"), or the
same file at a git revision (--rev, built as a separate extension), or the
wgmma prefill kernel the server runs for 9-32 tokens today ("prefill").

    bench_moe_m32.py --m 32 --kernel decode|prefill [--rev HEAD] [--cells N]

JSON lines on stdout; run NUMA-bound on the GPU's Grace node.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BF = load("bench_fmt_m32", EXP / "2026-10-08-moe-cost-table/bench_fmt_distinct.py")
HIDDEN, TOPK, GLOBAL = 6144, 8, 384


def routing(m, h, c, pool_hot, pool_cold, ntok_p, gen, device):
    hot_map = torch.full((GLOBAL,), -1, dtype=torch.int32)
    cold_map = torch.full((GLOBAL,), -1, dtype=torch.int32)
    hot_map[:h] = torch.randperm(pool_hot, generator=gen)[:h].to(torch.int32)
    cold_map[100:100 + c] = torch.randperm(pool_cold, generator=gen)[:c].to(torch.int32)
    ids = torch.full((m, TOPK), -1, dtype=torch.int32)
    fill = torch.zeros(m, dtype=torch.int64)
    for e in list(range(h)) + list(range(100, 100 + c)):
        n = 1 + int(torch.multinomial(ntok_p, 1, generator=gen))
        order = torch.randperm(m, generator=gen).tolist()
        for t in order:
            if n == 0:
                break
            if fill[t] < TOPK:
                ids[t, fill[t]] = e
                fill[t] += 1
                n -= 1
    remote = 200
    for t in range(m):
        while fill[t] < TOPK:
            ids[t, fill[t]] = remote % (GLOBAL - 200) + 200
            remote += 1
            fill[t] += 1
    w = torch.rand((m, TOPK), generator=gen)
    return ids.to(device), w.to(device), hot_map.to(device), cold_map.to(device)


def use_revision(rev: str):
    """Point tiered_decode at the kernel source of a git revision, built as
    its own extension (VLLM_TIERED_DECODE_DEFINES renames the build)."""
    import vllm.model_executor.layers.fused_moe.tiered_decode as td

    # compute nodes have no git: a pre-extracted tiered_decode_<rev>.cu here
    saved = HERE / f"tiered_decode_{rev}.cu"
    src = saved.read_text() if saved.exists() else subprocess.run(
        ["git", "-C", str(REPO), "show",
         f"{rev}:vllm/model_executor/layers/fused_moe/tiered_decode/tiered_decode.cu"],
        check=True, capture_output=True, text=True).stdout
    d = Path(tempfile.mkdtemp(prefix=f"td-{rev}-", dir=os.environ.get("TMPDIR")))
    (d / "tiered_decode.cu").write_text(src)
    os.environ["VLLM_TIERED_DECODE_DEFINES"] = f"TD_REV_{rev.replace('~', '_')}"
    td._HERE = d
    td._extension.cache_clear()
    td._workspace.cache_clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--kernel", choices=("decode", "prefill"), required=True)
    ap.add_argument("--rev")
    ap.add_argument("--cells", type=int, default=0, help="first N cells only")
    ap.add_argument("--numa-node", type=int, default=0)
    ap.add_argument("--pool-hot", type=int, default=64)
    ap.add_argument("--pool-cold", type=int, default=40)
    a = ap.parse_args()
    grid = json.loads((HERE / "grid-m8-16-32.json").read_text())[str(a.m)]
    hist = grid["all ranks"]["hist"]
    cells = [tuple(c) for c in grid["all ranks"]["cells"]]
    if a.cells:
        cells = cells[: a.cells]
    ntok = torch.tensor(grid["ntok"][1:a.m + 1], dtype=torch.float)
    ntok_p = ntok / ntok.sum()
    if a.rev:
        use_revision(a.rev)
    device = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(a.m)
    pool_hot = max(a.pool_hot, max(h for h, _ in cells) + 1)
    pool_cold = max(a.pool_cold, max(c for _, c in cells) + 1)
    hot, _ = BF.tier("int4", 16, device, False, a.numa_node, gen, pool_hot)
    cold, keep = BF.tier("int4", 8, device, True, a.numa_node, gen, pool_cold)
    x = (torch.randn((a.m, HIDDEN), generator=gen) * 0.3).to(torch.bfloat16).to(device)
    if a.kernel == "decode":
        from vllm.model_executor.layers.fused_moe.tiered_decode import tiered_decode_moe

        def call(c):
            return tiered_decode_moe(x, c[0], c[1], c[2], c[3], hot, cold)
    else:
        from vllm.model_executor.layers.fused_moe.tiered_prefill import (
            prefill_scale_exponent,
            tiered_prefill_moe,
        )

        exp = prefill_scale_exponent(hot, cold)

        def call(c):
            return tiered_prefill_moe(x, c[0], c[1], c[2], c[3], hot, cold, exp)
    for h, c in cells:
        calls = [routing(a.m, h, c, pool_hot, pool_cold, ntok_p, gen, device)
                 for _ in range(20)]
        for cl in calls[:2]:
            call(cl)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(20):
                call(calls[i])
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(10):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(5):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e) * 1000 / 100)
        print(json.dumps({"m": a.m, "kernel": a.kernel + (f"@{a.rev}" if a.rev else ""),
                          "hot": h, "cold": c, "weight": hist[h][c], "us": round(best, 1)}),
              flush=True)
    del keep


if __name__ == "__main__":
    sys.exit(main())
