"""HBM access-granularity bandwidth probe (mem-timing research, probe1).

Measures effective HBM bandwidth for streaming patterns and for random
gathers at 2 B, 256 B, and 64 KiB granularity. Streaming approximates
weight-traffic GEMV; fine-grained random gathers approximate KV-cache and
MoE-routing reads, which are the patterns whose cost is dominated by DRAM
row state (activation/precharge) rather than raw sector throughput.
"""

import argparse
import json
import os
import time

import torch

GiB = 1024**3


def bench(fn, bytes_moved, reps, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return bytes_moved * reps / dt / GiB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dev = torch.device("cuda:0")
    res = {
        "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
        "device": torch.cuda.get_device_name(dev),
        "torch": torch.__version__,
        "patterns": {},
    }

    n = 6 * GiB
    a = torch.empty(n, dtype=torch.uint8, device=dev)
    b = torch.empty(n, dtype=torch.uint8, device=dev)
    res["patterns"]["copy_6g_rw"] = bench(lambda: b.copy_(a), 2 * n, 10)
    res["patterns"]["fill_6g_w"] = bench(lambda: a.fill_(1), n, 10)
    del a, b
    s = torch.empty(3 * 2**30, dtype=torch.bfloat16, device=dev)
    res["patterns"]["sum_bf16_6g_r"] = bench(lambda: s.sum(), 6 * GiB, 10)
    del s

    src = torch.empty(4 * 2**30, dtype=torch.bfloat16, device=dev)  # 8 GiB
    elem = src.element_size()
    idx = torch.randint(0, src.numel(), (2**28,), device=dev)
    dst = torch.empty(2**28, dtype=torch.bfloat16, device=dev)  # 512 MiB
    moved = idx.numel() * elem * 2
    res["patterns"]["gather_2b"] = bench(
        lambda: torch.index_select(src, 0, idx, out=dst), moved, 20)
    del idx, dst

    srcv = src.view(-1, 128)  # 256 B rows
    ridx = torch.randint(0, srcv.shape[0], (2**21,), device=dev)
    rdst = torch.empty(2**21, 128, dtype=torch.bfloat16, device=dev)
    moved = ridx.numel() * 128 * elem * 2
    res["patterns"]["gather_256b"] = bench(
        lambda: torch.index_select(srcv, 0, ridx, out=rdst), moved, 20)
    del ridx, rdst

    srcv = src.view(-1, 32768)  # 64 KiB rows
    ridx = torch.randint(0, srcv.shape[0], (2**13,), device=dev)
    rdst = torch.empty(2**13, 32768, dtype=torch.bfloat16, device=dev)
    moved = ridx.numel() * 32768 * elem * 2
    res["patterns"]["gather_64kib"] = bench(
        lambda: torch.index_select(srcv, 0, ridx, out=rdst), moved, 20)

    stream = res["patterns"]["copy_6g_rw"]
    res["ratios_vs_copy"] = {
        k: round(v / stream, 4) for k, v in res["patterns"].items()
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nprobe1 results -> {args.out}")
    for k, v in res["patterns"].items():
        print(f"{k:18s} {v:8.1f} GiB/s   {res['ratios_vs_copy'][k]:6.3f} x copy")


if __name__ == "__main__":
    main()
