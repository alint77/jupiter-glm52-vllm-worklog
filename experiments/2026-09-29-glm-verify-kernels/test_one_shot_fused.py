#!/usr/bin/env python3
"""Fused one-shot DCP ops against the unfused one-shot path they replace, 4 GPUs.

    torchrun --nproc-per-node 4 test_one_shot_fused.py

* all_gather_cat(nope, pe) vs all_gather(torch.cat([nope, pe], -1), 1), with the
  MLA decode strides (nope is a transposed bmm output, pe a slice of q): exact.
* lse_reduce_scatter(out, lse) on the transposed [H, T] LSE view, unmasked
  (empty shards +inf, as FlashMLA returns them) vs mask_empty_dcp_lse +
  all_gather(lse) + correct_attn_out + reduce_scatter (production before the
  fusion), including -inf / NaN / +inf LSEs: bitwise agreement, max diff.

Eager and CUDA-graph-captured, interleaved with custom all-reduces, then graph
timings of 78 calls each (one per layer), fused vs unfused.
"""

import os

import torch

from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import mask_empty_dcp_lse
from vllm.v1.attention.ops.common import CPTritonContext, correct_attn_out

T, H_LOCAL, L, P, D = 8, 16, 512, 64, 512


def main() -> None:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world, rank, "env://", rank, "nccl")
    ensure_model_parallel_initialized(world, 1)
    tp = get_tp_group()
    from vllm.distributed.device_communicators.one_shot import OneShotCollectives

    ca = tp.device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom all-reduce unavailable"
    os_ = OneShotCollectives(ca)
    dev = torch.device("cuda", rank)
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)
    H = H_LOCAL * world

    def cases(special: bool):
        nbl = torch.randn((H_LOCAL, T, L), generator=gen, device=dev).to(torch.bfloat16)
        q = torch.randn((T, H_LOCAL, 192 + P), generator=gen, device=dev).to(
            torch.bfloat16
        )
        out = torch.randn((T, H, D), generator=gen, device=dev).to(torch.bfloat16)
        # FlashMLA returns lse as [H, T]; the combine gets the transposed view
        lse = (torch.randn((H, T), generator=gen, device=dev) * 4).t()
        if special:  # empty shards and all-empty heads, as in real decode
            lse[0, :4] = -float("inf")
            lse[1, 4:8] = float("nan")
            lse[2, 8:12] = float("inf")
            if rank < 2:
                lse[3, :] = float("inf")  # empty shard: the kernel's +inf row
            lse[4, 16:32] = -float("inf")  # all ranks: head fully masked
            lse[5, :] = float("inf")  # every shard empty for this token
        ar = torch.randn((T, 6144), generator=gen, device=dev).to(torch.bfloat16)
        return nbl, q, out, lse, ar

    def views(nbl, q):
        return nbl.transpose(0, 1), q[..., 192:]

    def fused(nbl, q, out, lse, ar):
        nope, pe = views(nbl, q)
        return (os_.all_gather_cat(nope, pe), ca.custom_all_reduce(ar),
                os_.lse_reduce_scatter(out, lse, True), ca.custom_all_reduce(ar))

    def unfused(nbl, q, out, lse, ar):
        nope, pe = views(nbl, q)
        gq = os_.all_gather(torch.cat([nope, pe], dim=-1), 1)
        a1 = ca.custom_all_reduce(ar)
        # production before the fusion: mask empty shards, then gather
        valid = (~torch.isposinf(lse).all(dim=1)).to(torch.int32)
        lse = mask_empty_dcp_lse(lse, valid).contiguous()
        lses = os_.all_gather(lse, 0).reshape(world, T, H)
        corrected, _ = correct_attn_out(out.clone(), lses, rank, CPTritonContext(), True)
        return gq, a1, os_.reduce_scatter(corrected, 1), ca.custom_all_reduce(ar)

    def compare(label, got, want):
        assert got[0] is not None and got[2] is not None, f"{label}: fused op fell back"
        torch.testing.assert_close(got[0], want[0], atol=0, rtol=0,
                                   msg=f"{label} gather_cat")
        g, w = got[2].float(), want[2].float()
        same = (g == w) | (g.isnan() & w.isnan())
        diff = (g - w).abs().nan_to_num(0).max().item()
        frac = torch.tensor([same.float().mean().item()], device=dev)
        worst = torch.tensor([diff, (~same).sum().item()], device=dev)
        torch.distributed.all_reduce(frac, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(worst, op=torch.distributed.ReduceOp.MAX)
        if rank == 0:
            print(f"{label}: lse_rs bitwise-equal {frac.item():.6f} (worst rank "
                  f"{worst[1].item():.0f} elements differ), max |diff| "
                  f"{worst[0].item():.3g}", flush=True)
        if (~same).any():
            i = (~same).nonzero()[0].tolist()
            print(f"  rank {rank} first diff at {i}: fused {g[tuple(i)].item()!r} "
                  f"ref {w[tuple(i)].item()!r}", flush=True)
        assert torch.equal(g.isnan(), w.isnan()), f"{label}: NaN pattern differs"
        torch.testing.assert_close(g.nan_to_num(0), w.nan_to_num(0), atol=1e-2,
                                   rtol=1e-2, msg=f"{label} lse_rs")

    for i in range(4):
        inputs = cases(special=i % 2 == 1)
        compare(f"eager {i}", fused(*inputs), unfused(*inputs))

    inputs = cases(special=True)
    with graph_capture(device=dev) as gctx:
        torch.cuda.synchronize()
        gf, gu = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(gf, stream=gctx.stream):
            got = fused(*inputs)
        with torch.cuda.graph(gu, stream=gctx.stream):
            want = unfused(*inputs)
    for i in range(2):
        for dst, src in zip(inputs, cases(special=i == 0)):
            dst.copy_(src)
        gf.replay()
        gu.replay()
        torch.cuda.synchronize()
        compare(f"graph {i}", got, want)

    # timing: 78 layers' worth, captured
    nbl, q, out, lse, _ = cases(special=False)
    valid_all = torch.ones(T, dtype=torch.int32, device=dev)
    nope, pe = views(nbl, q)
    ops = {
        "q: cat + gather": lambda: os_.all_gather(torch.cat([nope, pe], -1), 1),
        "q: gather_cat": lambda: os_.all_gather_cat(nope, pe),
        "combine: mask + gather + correct + rs": lambda: os_.reduce_scatter(
            correct_attn_out(
                out,
                os_.all_gather(
                    mask_empty_dcp_lse(lse, valid_all).contiguous(), 0
                ).reshape(world, T, H),
                rank,
                CPTritonContext(),
                True,
            )[0],
            1,
        ),
        "combine: lse_reduce_scatter": lambda: os_.lse_reduce_scatter(out, lse, True),
    }
    graphs = {}
    with graph_capture(device=dev) as gctx:
        for label, op in ops.items():
            op()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=gctx.stream):
                for _ in range(78):
                    op()
            graphs[label] = g
    for label, g in graphs.items():
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        if rank == 0:
            print(f"{label:34s} {s.elapsed_time(e) / 50 * 1000 / 78:6.2f} us per layer",
                  flush=True)
    if rank == 0:
        print("ONE-SHOT FUSED OK", flush=True)


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        main()
