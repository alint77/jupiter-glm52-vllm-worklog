# SPDX-License-Identifier: Apache-2.0
"""Scale repro: many interleaved DCP collectives in ONE captured graph.

The production FULL graph captures ~78 layers x (indexer all-gather, query
all-gather, LSE all-gather, reduce-scatter) on the DCP comm plus custom-AR
on TP. Single-collective captures pass; this tests N-at-scale.

Run: srun --ntasks=4 python pynccl_capture_repro3.py [num_layers]
"""

import os
import sys

import torch


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    layers = int(sys.argv[1]) if len(sys.argv) > 1 else 78
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        get_dcp_group,
        get_tp_group,
        graph_capture as vllm_graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
    )

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
    log(rank, f"parallel state ready; simulating {layers} layers")

    hidden = torch.randn(4, 6144, dtype=torch.bfloat16, device=device)
    q = torch.randn(4, 16, 576, dtype=torch.bfloat16, device=device)
    packed = torch.randn(4, 2048, 2, dtype=torch.float32, device=device)
    lse = torch.randn(4, 64, dtype=torch.float32, device=device)
    out64 = torch.randn(4, 64, 512, dtype=torch.bfloat16, device=device)
    w = torch.randn(6144, 6144, dtype=torch.bfloat16, device=device)

    def one_layer():
        tp.all_reduce(hidden)
        dcp.all_gather(packed, dim=1)  # indexer candidate exchange
        gq = dcp.all_gather(q, dim=1)  # query head fold
        h = hidden @ w  # stand-in compute
        dcp.all_gather(lse, dim=0)  # lse exchange
        ro = dcp.reduce_scatter(out64, dim=1)  # output scatter
        tp.all_reduce(hidden)
        return gq, h, ro

    def forward():
        for _ in range(layers):
            one_layer()

    for _ in range(2):
        forward()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    log(rank, "eager warmup OK")

    graph = torch.cuda.CUDAGraph()
    with vllm_graph_capture(device=device) as ctx:
        with torch.cuda.graph(graph, stream=ctx.stream):
            forward()
    log(rank, "capture OK")
    torch.distributed.barrier()
    graph.replay()
    torch.cuda.synchronize()
    log(rank, "replay OK")
    torch.distributed.barrier()
    log(rank, "DONE")


if __name__ == "__main__":
    main()
