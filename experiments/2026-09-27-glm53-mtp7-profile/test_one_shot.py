#!/usr/bin/env python3
"""One-shot all-gather / reduce-scatter against torch.distributed, 4 GPUs.

    torchrun --nproc-per-node 4 test_one_shot.py

Eager and CUDA-graph-captured, at DCP decode shapes (8 tokens: query gather
along heads, LSE gather along tokens, output reduce-scatter along heads),
interleaved with custom all-reduces so the shared barrier flags are
exercised in both orders. Exits non-zero on any mismatch.
"""

import os

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
)


def reference_gather(x: torch.Tensor, dim: int, group) -> torch.Tensor:
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, x.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def reference_rs(x: torch.Tensor, dim: int, group) -> torch.Tensor:
    total = x.float().clone()
    dist.all_reduce(total, group=group)
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    return total.chunk(world, dim=dim)[rank].to(x.dtype)


def main() -> None:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world, rank, "env://", rank, "nccl")
    ensure_model_parallel_initialized(world, 1)
    tp = get_tp_group()
    group = tp.device_group
    from vllm.distributed.device_communicators.one_shot import OneShotCollectives

    ca = tp.device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom all-reduce unavailable"
    os_ = OneShotCollectives(ca)
    dev = torch.device("cuda", rank)
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)

    def cases():
        q = torch.randn((8, 16, 576), generator=gen, device=dev).to(torch.bfloat16)
        lse = torch.randn((8, 64), generator=gen, device=dev)
        out = torch.randn((8, 64, 512), generator=gen, device=dev).to(torch.bfloat16)
        out32 = torch.randn((8, 64, 512), generator=gen, device=dev)
        ar = torch.randn((8, 6144), generator=gen, device=dev).to(torch.bfloat16)
        return q, lse, out, out32, ar

    def run(q, lse, out, out32, ar):
        return (os_.all_gather(q, 1), ca.custom_all_reduce(ar), os_.all_gather(lse, 0),
                os_.reduce_scatter(out, 1), os_.reduce_scatter(out32, 1),
                ca.custom_all_reduce(ar))

    def check(label, got, inputs):
        q, lse, out, out32, ar = inputs
        want = (reference_gather(q, 1, group), None, reference_gather(lse, 0, group),
                reference_rs(out, 1, group), reference_rs(out32, 1, group), None)
        for name, g, w in zip(("q gather", "ar", "lse gather", "out rs bf16",
                               "out rs fp32", "ar"), got, want):
            assert g is not None, f"{label} {name}: fell back to NCCL"
            if w is None:
                continue
            if "rs" in name:  # sums: rank order differs from NCCL's
                tol = 2e-2 if g.dtype == torch.bfloat16 else 1e-5
                torch.testing.assert_close(g.float(), w.float(), atol=tol, rtol=tol)
            else:  # gathers are copies: exact
                torch.testing.assert_close(g, w, atol=0, rtol=0, msg=f"{label} {name}")

    # eager, several rounds so staging through the registered buffer is reused
    for i in range(3):
        inputs = cases()
        check(f"eager {i}", run(*inputs), inputs)

    # captured: replay twice with fresh contents in the same input tensors
    inputs = cases()
    with graph_capture(device=dev) as ctx:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=ctx.stream):
            got = run(*inputs)
    for i in range(2):
        fresh = cases()
        for dst, src in zip(inputs, fresh):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        check(f"graph {i}", got, inputs)

    # timing: one captured graph of 78 query gathers
    q = cases()[0]
    with graph_capture(device=dev) as ctx:
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2, stream=ctx.stream):
            for _ in range(78):
                os_.all_gather(q, 1)
        g3 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g3, stream=ctx.stream):
            for _ in range(78):
                tp.all_gather(q, 1)
    for label, g in (("one-shot", g2), ("nccl", g3)):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(20):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        if rank == 0:
            print(f"q gather x78, {label}: {s.elapsed_time(e) / 20 * 1000 / 78:.2f} us each",
                  flush=True)
    if rank == 0:
        print("ONE-SHOT OK", flush=True)


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        main()
