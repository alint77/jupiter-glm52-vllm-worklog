# SPDX-License-Identifier: Apache-2.0
"""Staged 4-rank repro: grow the in-graph DCP all-gather toward vLLM's
production composition until capture dies.

Stages (each prints per-rank progress; the last passed stage brackets the
crashing ingredient):
  1. vLLM parallel state (world/TP/DCP comms), DCP all-gather in plain capture
  2. same inside vLLM's graph_capture() context (side stream + ca capture)
  3. TP custom all-reduce + DCP all-gather in the same graph
  4. aux-stream fork/join (tiered-style) + stage 3 in the same graph

Run: srun --ntasks=4 python pynccl_capture_repro2.py
"""

import os

import torch


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def capture(fn, use_vllm_ctx, device):
    graph = torch.cuda.CUDAGraph()
    if use_vllm_ctx:
        from vllm.distributed.parallel_state import graph_capture as vllm_graph_capture

        with vllm_graph_capture(device=device) as ctx:
            with torch.cuda.graph(graph, stream=ctx.stream):
                fn()
    else:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def main():
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    # All GPUs visible per task (production mp-executor layout): custom-AR
    # P2P checks need every device enumerable.
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    from vllm.distributed.parallel_state import (
        get_dcp_group,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm.config import VllmConfig, set_current_vllm_config

    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank, backend="nccl"
    )
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(
            tensor_model_parallel_size=world,
            decode_context_model_parallel_size=world,
        )
    dcp = get_dcp_group()
    tp = get_tp_group()
    log(rank, "vllm parallel state ready (tp + dcp comms)")

    q = torch.randn(4, 16, 576, dtype=torch.bfloat16, device=device)
    hidden = torch.randn(4, 6144, dtype=torch.bfloat16, device=device)

    def dcp_ag():
        return dcp.all_gather(q, dim=1)

    def tp_ar_and_dcp_ag():
        tp.all_reduce(hidden)
        return dcp.all_gather(q, dim=1)

    aux = torch.cuda.Stream()

    def aux_fork_tp_ar_dcp_ag():
        main_stream = torch.cuda.current_stream()
        aux.wait_stream(main_stream)
        with torch.cuda.stream(aux):
            cold = hidden * 2.0
        tp.all_reduce(hidden)
        out = dcp.all_gather(q, dim=1)
        main_stream.wait_stream(aux)
        hidden.add_(cold)
        return out

    for _ in range(3):
        dcp_ag()
        tp_ar_and_dcp_ag()
        aux_fork_tp_ar_dcp_ag()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    log(rank, "eager warmup OK")

    capture(dcp_ag, use_vllm_ctx=False, device=device)
    log(rank, "stage1 OK: dcp allgather, plain capture")
    torch.distributed.barrier()

    capture(dcp_ag, use_vllm_ctx=True, device=device)
    log(rank, "stage2 OK: dcp allgather, vllm graph_capture ctx")
    torch.distributed.barrier()

    capture(tp_ar_and_dcp_ag, use_vllm_ctx=True, device=device)
    log(rank, "stage3 OK: tp custom-AR + dcp allgather, same graph")
    torch.distributed.barrier()

    capture(aux_fork_tp_ar_dcp_ag, use_vllm_ctx=True, device=device)
    log(rank, "stage4 OK: aux-stream fork + tp AR + dcp allgather")
    torch.distributed.barrier()
    log(rank, "DONE")


if __name__ == "__main__":
    main()
