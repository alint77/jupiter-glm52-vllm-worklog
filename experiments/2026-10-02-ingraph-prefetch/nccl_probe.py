"""The prefill step's collectives at their real sizes (2464 tokens, GLM-5.3,
DCP4/TP4), timed per protocol: MoE/attention all-reduce [2464, 6144] bf16,
DCP q all-gather [2464, 16, 576] per rank, DCP output reduce-scatter
[2464, 64, 512]. One process per GPU; NCCL_PROTO / NCCL_ALGO from the env.
Usage: python nccl_probe.py  (spawns 4 ranks)"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

T = 2464


def run(rank):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29531",
                            rank=rank, world_size=4)
    dev = torch.device("cuda", rank)
    ar = torch.randn(T, 6144, dtype=torch.bfloat16, device=dev)
    ag_in = torch.randn(T * 16 * 576, dtype=torch.bfloat16, device=dev)
    ag_out = torch.empty(4 * ag_in.numel(), dtype=torch.bfloat16, device=dev)
    rs_in = torch.randn(T * 64 * 512, dtype=torch.bfloat16, device=dev)
    rs_out = torch.empty(rs_in.numel() // 4, dtype=torch.bfloat16, device=dev)
    ops = {
        "all_reduce 30.3 MB": lambda: dist.all_reduce(ar),
        "all_gather 45.4 MB/rank": lambda: dist.all_gather_into_tensor(ag_out, ag_in),
        "reduce_scatter 161.5 MB": lambda: dist.reduce_scatter_tensor(rs_out, rs_in),
    }
    res = {}
    for name, fn in ops.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize(); dist.barrier()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(20):
            fn()
        b.record(); torch.cuda.synchronize()
        res[name] = a.elapsed_time(b) * 1000 / 20
    if rank == 0:
        tag = f"PROTO={os.environ.get('NCCL_PROTO', 'default')} ALGO={os.environ.get('NCCL_ALGO', 'default')}"
        print(tag + "  " + "  ".join(f"{k}: {v:.0f} us" for k, v in res.items()), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, nprocs=4)
