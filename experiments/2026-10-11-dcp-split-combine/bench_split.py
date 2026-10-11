#!/usr/bin/env python3
"""FlashMLA sparse decode + DCP combine: separate vs fused split combine.

    torchrun --nproc-per-node 4 bench_split.py [--tokens 8,32,64]

Per layer, at the production decode shape (q [1, T, 64, 576], fp8_ds_mla KV,
768-wide index rows with ~512 live, a separate KV region per layer so the
gather misses L2 as in serving):
  separate  FlashMLA sparse_decode_fwd (main + split-KV combine), then
            one_shot lse_reduce_scatter                (production today)
  fused     FlashMLA sparse_decode_fwd_partial (main only), then
            one_shot split_lse_reduce_scatter
Bitwise check first, then graph-captured timing of --layers layers (max over
ranks) and profiler kernel means.
"""

import argparse
import os

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
)

H, WIDTH, REGIONS, BLOCKS = 64, 768, 8, 1600


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="8,32,64")
    ap.add_argument("--layers", type=int, default=78)
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world, rank, "env://", rank, "nccl")
    ensure_model_parallel_initialized(world, 1)
    import vllm._flashmla_C  # noqa: F401
    from vllm.distributed.device_communicators.one_shot import OneShotCollectives

    fmla = torch.ops._flashmla_C
    ca = get_tp_group().device_communicator.ca_comm
    coll = OneShotCollectives(ca)
    dev = torch.device("cuda", rank)
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)
    n = BLOCKS * 64

    def region():
        nope = (torch.rand((n, 512), generator=gen, device=dev) * 4 - 2).to(
            torch.float8_e4m3fn)
        scale = torch.rand((n, 4), generator=gen, device=dev) * 0.5 + 0.05
        rope = torch.randn((n, 64), generator=gen, device=dev).to(torch.bfloat16)
        return torch.cat([nope.view(torch.uint8), scale.view(torch.uint8),
                          rope.view(torch.uint8)], 1).view(BLOCKS, 64, 1, 656)

    kvs = [region() for _ in range(REGIONS)]

    def inputs(T):
        q = (torch.randn((1, T, H, 576), generator=gen, device=dev) / 8).bfloat16()
        idx = torch.randint(0, n, (1, T, WIDTH), generator=gen, device=dev).int()
        live = torch.randint(470, 540, (T, 1), generator=gen, device=dev)
        idx = idx.masked_fill(torch.arange(WIDTH, device=dev) >= live, -1)
        return q, idx

    def separate(q, kv, idx, sched):
        out, lse, *_ = fmla.sparse_decode_fwd(
            q, kv, idx, None, None, *sched, None, None, None, 512, 0.07, None)
        return coll.lse_reduce_scatter(out[0], lse.transpose(1, 2)[0], True)

    def fused(q, kv, idx, sched):
        out, lse, oa, la, _, splits = fmla.sparse_decode_fwd_partial(
            q, kv, idx, None, None, *sched, None, None, None, 512, 0.07, None)
        return coll.split_lse_reduce_scatter(oa, la, splits, out[0],
                                             lse.transpose(1, 2)[0])

    def main_only(q, kv, idx, sched):
        fmla.sparse_decode_fwd_partial(
            q, kv, idx, None, None, *sched, None, None, None, 512, 0.07, None)

    def main_combine(q, kv, idx, sched):
        fmla.sparse_decode_fwd(
            q, kv, idx, None, None, *sched, None, None, None, 512, 0.07, None)

    arms = {"separate": separate, "fused": fused, "main": main_only,
            "main+combine": main_combine}
    for T in [int(t) for t in args.tokens.split(",")]:
        q, idx = inputs(T)
        _, _, sm, sp = fmla.sparse_decode_fwd(
            q, kvs[0], idx, None, None, None, None, None, None, None, 512, 0.07,
            None)
        sched = (sm, sp)
        a, b = separate(q, kvs[0], idx, sched), fused(q, kvs[0], idx, sched)
        torch.cuda.synchronize()
        ok = torch.tensor([int(torch.equal(a.view(torch.int16),
                                           b.view(torch.int16)))], device=dev)
        torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
        splits = int((sp[1] - sp[0]).item())
        graphs = []
        with graph_capture(device=dev) as gctx:
            for fn in arms.values():
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=gctx.stream):
                    for layer in range(args.layers):
                        fn(q, kvs[layer % REGIONS], idx, sched)
                graphs.append(g)
        times = []
        for g in graphs:
            g.replay()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            for _ in range(20):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) / 20 / args.layers * 1000)
        out0, lse0, *_ = fmla.sparse_decode_fwd(
            q, kvs[0], idx, None, None, *sched, None, None, None, 512, 0.07, None)
        g = torch.cuda.CUDAGraph()
        with graph_capture(device=dev) as gctx, torch.cuda.graph(g, stream=gctx.stream):
            for layer in range(args.layers):
                coll.lse_reduce_scatter(out0[0], lse0.transpose(1, 2)[0], True)
        graphs.append(g)
        g.replay()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        for _ in range(20):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 20 / args.layers * 1000)
        res = torch.tensor(times, device=dev)
        torch.distributed.all_reduce(res, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.barrier()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for g in graphs:
                torch.distributed.barrier()
                g.replay()
                torch.cuda.synchronize()
        kern = {}
        for ev in prof.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                kern.setdefault(ev.name.split("(")[0][:60], []).append(ev.device_time)
        if rank == 0:
            print(f"T={T:3d} splits={splits:3d} bitwise={'OK' if ok.item() else 'FAIL'}"
                  f"  separate {res[0]:6.2f}  fused {res[1]:6.2f} us/layer"
                  f"  (saved {res[0] - res[1]:5.2f})", flush=True)
            print("   graph us/layer: " + "  ".join(
                f"{k} {v:.2f}" for k, v in zip([*arms, "lse_rs"], res.tolist())),
                flush=True)
            for name, d in sorted(kern.items()):
                print(f"   {name:60s} n={len(d):4d} mean {sum(d) / len(d):7.2f} us",
                      flush=True)
    if rank == 0:
        print("SPLIT BENCH DONE", flush=True)


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        main()
