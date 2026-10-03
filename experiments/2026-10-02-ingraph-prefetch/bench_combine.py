"""DCP prefill attention combine at the real shape (T 4089, 64 heads, 512,
bf16, 4 ranks): the fused one-shot combine vs the NCCL path (LSE all-gather,
correction, reduce-scatter). Kernel times from the torch profiler, rank 0."""
import os
import sys

import torch
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.getcwd())
T, H, D = 4089, 64, 512


def worker(rank, port):
    from tests.utils import init_test_distributed_environment
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        get_tp_group,
    )

    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(4, 1, rank, port)
    ensure_model_parallel_initialized(4, 1)
    from vllm.distributed.device_communicators.one_shot import OneShotCollectives
    from vllm.v1.attention.ops import common

    tp = get_tp_group()
    ca = tp.device_communicator.ca_comm
    os_ = OneShotCollectives(ca)
    os_.setup_prefill_buffer(4096 * H * (D * 2 + 4) + 256)
    dev = torch.device("cuda", rank)
    out = torch.randn((T, H, D), device=dev).to(torch.bfloat16)
    lse = torch.randn((H, T), device=dev).t()
    buf = os_.prefill_out(T, H, D, out.dtype)
    buf.copy_(out)

    def fused():
        os_._prefill_combine(buf, lse, True, head_major=True)

    def nccl():
        B = T
        head_major = out.new_empty((H, B, D))
        common._cp_lse_common(out, lse, tp, new_out=head_major.transpose(0, 1))
        tp.reduce_scatter(head_major, dim=0)

    for name, fn in (("fused", fused), ("nccl", nccl)):
        for _ in range(3):
            fn()
        torch.accelerator.synchronize()
        tp.barrier()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                fn()
            torch.accelerator.synchronize()
        if rank == 0:
            agg = {}
            for e in prof.events():
                if e.device_type.name == "CUDA":
                    k = e.name[:60]
                    agg[k] = agg.get(k, 0) + e.device_time / 10
            print(f"== {name}: " + ", ".join(f"{k} {v:.0f} us" for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:5]), flush=True)


if __name__ == "__main__":
    mp.spawn(worker, args=(str(29500 + os.getpid() % 1000),), nprocs=4)
