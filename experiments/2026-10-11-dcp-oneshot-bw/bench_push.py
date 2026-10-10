#!/usr/bin/env python3
"""Push-mode barrier-free DCP collectives vs the production pull kernels.

    torchrun --nproc-per-node 4 bench_push.py [--tokens 8,16,32,64]

1. Stress: a graph of 12 calls, gather_cat and lse_rs alternating, each on its
   own inputs and outputs, with rank- and call-dependent spin delays in front
   of each call (ranks drift apart, so a call's sends land while a peer is
   still in an earlier call). Every output bitwise vs the production kernels
   on the same inputs; replayed several times with fresh data.
2. Timing: graphs of 78 calls per (blocks, unroll) config, max over ranks,
   plus profiler kernel means.
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
SLOT = 1310720


def load_bw():
    from torch.utils.cpp_extension import load

    build = Path(os.environ["VLLM_CACHE_ROOT"]) / "torch_extensions" / "one_shot_bw"
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="one_shot_bw",
        sources=[str(HERE / "one_shot_bw.cu")],
        extra_include_paths=[str(CSRC), str(HERE)],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="8,16,32,64")
    ap.add_argument("--blocks", default="16,32,64,132")
    ap.add_argument("--unroll", default="2,4,8")
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--stress-rounds", type=int, default=20)
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world, rank, "env://", rank, "nccl")
    ensure_model_parallel_initialized(world, 1)
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm.distributed.device_communicators.one_shot import OneShotCollectives

    ca = get_tp_group().device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom all-reduce unavailable"
    prod = OneShotCollectives(ca)
    if rank == 0:
        load_bw()
    torch.distributed.barrier()
    bw = load_bw()
    dev = torch.device("cuda", rank)
    nbytes = bw.push_buffer_bytes(SLOT, world)
    ptrs = CustomAllreduce.create_shared_buffer(nbytes, group=ca.group)
    bw.view(ptrs[rank], nbytes).zero_()
    epoch = torch.zeros(2, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    bw.push_setup(ptrs, SLOT, epoch.data_ptr())
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)
    H = H_LOCAL * world

    def report(msg):
        if rank == 0:
            print(msg, flush=True)

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

    def push_gc(nbl, q):
        nope, pe = nbl.transpose(0, 1), q[..., 192:]
        T = nope.shape[0]
        dst = torch.empty((T, H, L + P), dtype=torch.bfloat16, device=dev)
        bw.push_all_gather_cat(nope, pe, dst, world, rank)
        return dst

    def push_lse(out, lse):
        T = out.shape[0]
        dst = torch.empty((T, H_LOCAL, D), dtype=torch.bfloat16, device=dev)
        bw.push_lse_reduce_scatter(out, lse, dst, True, world, rank)
        return dst

    def prod_gc(nbl, q):
        return prod.all_gather_cat(nbl.transpose(0, 1), q[..., 192:])

    def prod_lse(out, lse):
        return prod.lse_reduce_scatter(out, lse, True)

    def agree(got, want, label):
        ok = torch.tensor([int(torch.equal(got.view(torch.int16),
                                           want.view(torch.int16)))], device=dev)
        torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
        assert ok.item() == 1, f"{label}: not bitwise equal"

    tokens = [int(t) for t in args.tokens.split(",")]
    blocks = [int(b) for b in args.blocks.split(",")]
    unrolls = [int(u) for u in args.unroll.split(",")]

    # 1. stress with drift
    for T in tokens:
        for nb in (blocks[0], blocks[-1]):
            bw.push_config(nb, unrolls[-1])
            ins = [cases(T, special=True) for _ in range(12)]
            with graph_capture(device=dev) as gctx:
                torch.cuda.synchronize()
                gp, gr = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(gp, stream=gctx.stream):
                    got = []
                    for i, (nbl, q, out, lse) in enumerate(ins):
                        torch.cuda._sleep(((rank * 7 + i * 3) % 11) * 2000)
                        got.append(push_gc(nbl, q) if i % 2 == 0 else push_lse(out, lse))
                with torch.cuda.graph(gr, stream=gctx.stream):
                    want = [prod_gc(nbl, q) if i % 2 == 0 else prod_lse(out, lse)
                            for i, (nbl, q, out, lse) in enumerate(ins)]
            for rnd in range(args.stress_rounds):
                for tensors in ins:
                    for dst, src in zip(tensors, cases(T, special=rnd % 2 == 0)):
                        dst.copy_(src)
                gp.replay()
                gr.replay()
                torch.cuda.synchronize()
                for i, (g, w) in enumerate(zip(got, want)):
                    agree(g, w, f"stress T={T} blocks={nb} round {rnd} call {i}")
            # eager too
            for i, (nbl, q, out, lse) in enumerate(ins[:4]):
                g = push_gc(nbl, q) if i % 2 == 0 else push_lse(out, lse)
                w = prod_gc(nbl, q) if i % 2 == 0 else prod_lse(out, lse)
                torch.cuda.synchronize()
                agree(g, w, f"eager T={T} blocks={nb} call {i}")
        report(f"T={T}: push bitwise equal to production "
               f"({args.stress_rounds} drifted rounds x 12 calls, eager)")

    # 2. timing
    for T in tokens:
        nbl, q, out, lse = cases(T, special=False)
        configs = [("gc", "prod", 0, 0), ("lse", "prod", 0, 0)]
        configs += [(k, "push", nb, u) for k in ("gc", "lse")
                    for nb in blocks for u in unrolls]
        graphs = []
        with graph_capture(device=dev) as gctx:
            for kind, impl, nb, u in configs:
                if impl == "push":
                    bw.push_config(nb, u)
                fn = {("gc", "prod"): lambda: prod_gc(nbl, q),
                      ("lse", "prod"): lambda: prod_lse(out, lse),
                      ("gc", "push"): lambda: push_gc(nbl, q),
                      ("lse", "push"): lambda: push_lse(out, lse)}[(kind, impl)]
                fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=gctx.stream):
                    for _ in range(args.layers):
                        fn()
                graphs.append(g)
        times = []
        for g in graphs:
            g.replay()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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
            for g in graphs[:2]:
                torch.distributed.barrier()
                g.replay()
                torch.cuda.synchronize()
        kern = {}
        for ev in prof.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                kern.setdefault(ev.name.split("(")[0], []).append(ev.device_time)
        if rank == 0:
            for (kind, impl, nb, u), t in zip(configs, res.tolist()):
                tag = "prod" if impl == "prod" else f"push b={nb} u={u}"
                print(f"T={T:3d} {kind:3s} {tag:16s} {t:7.2f} us/call", flush=True)
            for name, d in sorted(kern.items()):
                if "one_shot" in name:
                    print(f"   kernel {name[:70]:70s} mean {sum(d) / len(d):7.2f} us",
                          flush=True)
    report("PUSH BENCH DONE")


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        main()
