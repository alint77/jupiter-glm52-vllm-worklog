# SPDX-License-Identifier: Apache-2.0
"""Minimal 4-rank repro: pynccl all_gather inside CUDA graph capture.

Mirrors the production pattern the DCP port introduces (the tiered DCP1
config never captures an NCCL collective). Run with:
  srun --ntasks=4 python pynccl_capture_repro.py
"""

import os

import torch
import torch.distributed as dist


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group("gloo")
    # --gpus-per-task=1: each task sees exactly one GPU as device 0.
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    comm = PyNcclCommunicator(group=dist.group.WORLD, device=device)
    assert not comm.disabled
    log(rank, "pynccl ready")

    inp = torch.randn(4 * 16 * 576, dtype=torch.bfloat16, device=device)
    out = torch.empty(world * inp.numel(), dtype=torch.bfloat16, device=device)

    for _ in range(3):
        comm.all_gather(out, inp)
    torch.cuda.synchronize()
    dist.barrier()
    log(rank, "eager all_gather OK")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            comm.all_gather(out, inp)
    log(rank, "capture OK")
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    log(rank, "replay OK")
    dist.barrier()
    log(rank, "DONE")


if __name__ == "__main__":
    main()
