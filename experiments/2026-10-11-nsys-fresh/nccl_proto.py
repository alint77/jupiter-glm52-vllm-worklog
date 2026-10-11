"""NCCL all-gather / all-reduce at the prefill message sizes (4 GH200,
torch.distributed, CUDA-graph-free, median of 30), per NCCL_PROTO setting
given in the environment. Shapes per rank:
  q all-gather: T x 16 heads x 576 bf16 (DCP q gather), gathered to 4x
  all-reduce:   T x 6144 bf16 (TP all-reduce)

    torchrun --nproc-per-node 4 nccl_proto.py
"""
import os
import statistics as st

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
r = dist.get_rank()
torch.cuda.set_device(r)
out = []
for T in (512, 1024, 2048, 4096):
    q = torch.randn(T * 16 * 576, dtype=torch.bfloat16, device="cuda")
    qa = torch.empty(4 * q.numel(), dtype=torch.bfloat16, device="cuda")
    x = torch.randn(T * 6144, dtype=torch.bfloat16, device="cuda")
    res = {}
    for name, fn, nbytes in (
        ("allgather", lambda: dist.all_gather_into_tensor(qa, q), 3 * q.numel() * 2),
        ("allreduce", lambda: dist.all_reduce(x), 2 * 0.75 * x.numel() * 2),
    ):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        m = st.median(ts)
        res[name] = (m, nbytes / m / 1e3)
    out.append((T, res))
if r == 0:
    p = os.environ.get("NCCL_PROTO", "default")
    for T, res in out:
        print(f"proto={p:8s} T={T:5d}  allgather {res['allgather'][0]:7.1f} us "
              f"({res['allgather'][1]:5.0f} GB/s in)  allreduce {res['allreduce'][0]:7.1f} us "
              f"({res['allreduce'][1]:5.0f} GB/s bus)", flush=True)
dist.destroy_process_group()
