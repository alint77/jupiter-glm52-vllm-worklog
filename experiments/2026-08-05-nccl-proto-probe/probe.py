"""Measure NCCL all-reduce bandwidth per protocol at the prefill message size.

Two questions, both unanswered by the A/B that tried to set NCCL_PROTO from the
shell and found the kernels unchanged:

1. Is NCCL_PROTO honoured at all on this stack? NCCL logs "NCCL_PROTO set by
   environment to ..." at init under NCCL_DEBUG=INFO, which settles it.
2. Would Simple even be faster? Production's 96 MiB all-reduce runs RING_LL at
   a 70 GB/s bus rate. If Simple is no better here, the lever dies regardless of
   how the variable is plumbed.

Standalone torch.distributed, so nothing about vLLM's env handling is involved.
"""

import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

HIDDEN = 6144
CHUNK_TOKENS = 8192  # the production prefill chunk
WARMUP, ITERS = 10, 30


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    rows = []
    for tokens in (CHUNK_TOKENS, CHUNK_TOKENS * 2):
        x = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
        nbytes = x.numel() * x.element_size()
        for _ in range(WARMUP):
            dist.all_reduce(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(ITERS):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            dist.all_reduce(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000)  # s

        t = statistics.median(times)
        # Ring all-reduce moves 2(N-1)/N x S per rank.
        bus = nbytes * 2 * (world - 1) / world / t / 1e9
        rows.append({
            "tokens": tokens,
            "MiB": round(nbytes / 2**20, 1),
            "median_us": round(t * 1e6, 1),
            "algbw_GBs": round(nbytes / t / 1e9, 1),
            "busbw_GBs": round(bus, 1),
        })

    if rank == 0:
        print(json.dumps({
            "NCCL_PROTO": os.environ.get("NCCL_PROTO", "<unset>"),
            "NCCL_ALGO": os.environ.get("NCCL_ALGO", "<unset>"),
            "world": world,
            "results": rows,
        }))
        sys.stdout.flush()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
