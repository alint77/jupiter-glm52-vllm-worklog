#!/usr/bin/env python3
"""DCP4 one-shot decode collectives: current kernels vs more loads in flight.

    torchrun --nproc-per-node 4 bench_dcp.py [--tokens 8,16,32,64]

The production extension (one_shot.cu) is the reference; one_shot_bw.cu is a
copy with variant kernels picked by set_variant(gc, lse):
  gc  0 original, U>0 U loads in flight per thread (32-bit index math),
      -U the same with 64-bit index math
  lse 0 original, 1/2 one warp per (token, head) pair, 1 or 2 pairs per warp
Per token count: bitwise check of every variant against the production
kernel on MLA decode strides (special LSEs included), then graph-captured
timing of 78 calls (one per layer), kernel time from the torch profiler.
"""

import argparse
import os
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
)

HERE = Path(__file__).resolve().parent
CSRC = HERE.parents[2] / "csrc"
H_LOCAL, L, P, D = 16, 512, 64, 512
GC_VARIANTS = [0, 1, 2, 4, 8, 16, -8]
LSE_VARIANTS = [0, 1, 2]


def load_bw():
    from torch.utils.cpp_extension import load

    build = Path(os.environ["VLLM_CACHE_ROOT"]) / "torch_extensions" / "one_shot_bw"
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="one_shot_bw",
        sources=[str(HERE / "one_shot_bw.cu")],
        extra_include_paths=[str(CSRC)],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="8,16,32,64")
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--blocks", default="0",
                    help="probe: block counts for the variants (>36 unsafe, "
                    "timing only; exactness is checked at 0 only)")
    ap.add_argument("--il", type=int, default=0, help="interleave ranks (gc)")
    ap.add_argument("--gc", default=",".join(map(str, GC_VARIANTS)))
    ap.add_argument("--lse", default=",".join(map(str, LSE_VARIANTS)))
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world, rank, "env://", rank, "nccl")
    ensure_model_parallel_initialized(world, 1)
    from vllm.distributed.device_communicators.one_shot import OneShotCollectives

    ca = get_tp_group().device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom all-reduce unavailable"
    prod = OneShotCollectives(ca)
    if rank == 0:
        load_bw()
    torch.distributed.barrier()
    bw = load_bw()
    bw.set_interleave(args.il)
    dev = torch.device("cuda", rank)
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)
    H = H_LOCAL * world

    def cases(T, special):
        nbl = torch.randn((H_LOCAL, T, L), generator=gen, device=dev).bfloat16()
        q = torch.randn((T, H_LOCAL, 192 + P), generator=gen, device=dev).bfloat16()
        out = torch.randn((T, H, D), generator=gen, device=dev).bfloat16()
        lse = (torch.randn((H, T), generator=gen, device=dev) * 4).t()
        if special:
            lse[0, :4] = -float("inf")
            lse[1, 4:8] = float("nan")
            lse[2, 8:12] = float("inf")
            if rank < 2:
                lse[3, :] = float("inf")
            lse[4, 16:32] = -float("inf")
            lse[5, :] = float("inf")
        return nbl, q, out, lse

    def gc_op(ext, nope, pe, dst):
        if ext is None:
            return prod.all_gather_cat(nope, pe)
        ext.all_gather_cat(ca._ptr, nope, pe, dst, 0, 0)
        return dst

    def lse_op(ext, out, lse, dst):
        if ext is None:
            return prod.lse_reduce_scatter(out, lse, True)
        ext.lse_reduce_scatter(ca._ptr, out, lse, dst, True, 0, 0)
        return dst

    def report(msg):
        if rank == 0:
            print(msg, flush=True)

    for T in [int(t) for t in args.tokens.split(",")]:
        nbl, q, out, lse = cases(T, special=True)
        nope, pe = nbl.transpose(0, 1), q[..., 192:]
        blist = [int(b) for b in args.blocks.split(",")]
        variants = [("gc", None, None, 0)] + [
            ("gc", bw, int(v), b) for b in blist for v in args.gc.split(",")]
        variants += [("lse", None, None, 0)] + [
            ("lse", bw, int(v), b) for b in blist for v in args.lse.split(",")]
        outs, graphs = [], []
        with graph_capture(device=dev) as gctx:
            for kind, ext, v, nb in variants:
                gdst = torch.empty((T, H, L + P), dtype=torch.bfloat16, device=dev)
                ldst = torch.empty((T, H_LOCAL, D), dtype=torch.bfloat16, device=dev)
                if ext is not None:
                    ext.set_blocks(nb)
                    ext.set_variant(v if kind == "gc" else 0, v if kind == "lse" else 0)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=gctx.stream):
                    if kind == "gc":
                        o = gc_op(ext, nope, pe, gdst)
                    else:
                        o = lse_op(ext, out, lse, ldst)
                outs.append(o)
                gt = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gt, stream=gctx.stream):
                    for _ in range(args.layers):
                        if kind == "gc":
                            gc_op(ext, nope, pe, gdst)
                        else:
                            lse_op(ext, out, lse, ldst)
                graphs.append((g, gt))
        # exactness: fresh inputs each round, every graph replayed
        for rnd in range(3):
            for dst, src in zip((nbl, q, out, lse), cases(T, special=rnd != 1)):
                dst.copy_(src)
            for g, _ in graphs:
                g.replay()
            torch.cuda.synchronize()
            ref = {"gc": None, "lse": None}
            for (kind, ext, v, nb), o in zip(variants, outs):
                if ext is None:
                    ref[kind] = o.clone()
                    continue
                if nb:
                    continue
                r = ref[kind]
                same = torch.equal(o.view(torch.int16), r.view(torch.int16))
                ok = torch.tensor([int(same)], device=dev)
                torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
                assert ok.item() == 1, f"T={T} {kind} variant {v} round {rnd}: not bitwise equal"
        report(f"T={T}: all default-block variants bitwise equal to production (3 rounds)")
        # timing
        for _, gt in graphs:
            gt.replay()
        torch.cuda.synchronize()
        times = []
        for g, gt in graphs:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.distributed.barrier()
            torch.cuda.synchronize()
            s.record()
            for _ in range(20):
                gt.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) / 20 / args.layers * 1000)
        torch.distributed.barrier()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _, gt in graphs:
                torch.distributed.barrier()
                for _ in range(5):
                    gt.replay()
                torch.cuda.synchronize()
        kern = {}
        for ev in prof.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                kern.setdefault(ev.name, []).append(ev.device_time)
        # kernels in capture order: name -> durations; map variants by name
        names = {
            ("gc", None): "gather_cat_kernel",
            ("lse", None): "lse_reduce_scatter_kernel",
        }
        res = torch.tensor(times, device=dev)
        torch.distributed.all_reduce(res, op=torch.distributed.ReduceOp.MAX)
        if rank == 0:
            for (kind, ext, v, nb), t in zip(variants, res.tolist()):
                tag = "prod" if ext is None else f"v={v} b={nb}"
                print(f"T={T:3d} {kind:3s} {tag:12s} {t:7.2f} us/call (graph, max rank)",
                      flush=True)
            for name, d in sorted(kern.items()):
                short = name.split("(")[0][:70]
                print(f"   kernel {short:70s} n={len(d):5d} mean {sum(d) / len(d):7.2f} us",
                      flush=True)
    report("BENCH DONE")


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        main()
