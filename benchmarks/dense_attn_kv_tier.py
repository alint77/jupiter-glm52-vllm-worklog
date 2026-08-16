#!/usr/bin/env python3
"""KV-in-HBM vs KV-in-Grace for *dense* attention kernels.

The sparse (DSA) result in `2026-08-05-kv-grace-attn` is dominated by the access
pattern: a top-k of 2,048 rows scattered across a 20 GB cache, where Grace
saturates near 157 GB/s while HBM keeps scaling past 760. Dense attention reads
the KV contiguously instead, which is the regime C2C is actually good at, so the
answer should differ — and dense reads vastly more bytes, so it may lose anyway.

One layer per measurement, swept over context length, reporting achieved
bandwidth. Per-layer numbers scale linearly to a whole model.

Requires the caller to be NUMA-bound to the paired Grace node; the pinned
allocator audits placement but does not bind (see GraceAllocation).
"""

import argparse
import json
import statistics

import torch

from vllm.model_executor.offloader.grace import GraceAllocation

# GLM-5.2 MLA geometry, matching mla_cache_full_footprint.py
MLA_ENTRY_BYTES = 656
MLA_HEADS = 64
MLA_HEAD_DIM = 576
MLA_VALUE_DIM = 512
BLOCK_SIZE = 64

# Llama-3-70B-shaped GQA
GQA_Q_HEADS = 64
GQA_KV_HEADS = 8
GQA_HEAD_DIM = 128


def timed(fn, warmups: int, iters: int) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def alloc(nbytes: int, tier: str, device: torch.device, numa_node: int):
    """Return (tensor_uint8, keepalive) for one KV buffer in the given tier."""
    if tier == "hbm":
        return torch.zeros(nbytes, dtype=torch.uint8, device=device), None
    allocation = GraceAllocation.allocate_pinned(
        (nbytes,), torch.int8, device.index or 0, numa_node
    )
    allocation.cpu_tensor.zero_()
    placement = allocation.audit_numa(samples=32)
    if placement.local_fraction < 0.95:
        raise SystemExit(
            f"Grace allocation only {placement.local_fraction:.0%} local to node "
            f"{numa_node}; bind with numactl --membind"
        )
    return allocation.cuda_alias.view(torch.uint8), allocation


def bench_mla_dense(
    context: int, q_tokens: int, batch: int, tier: str, device, numa_node, args
):
    from vllm.v1.attention.ops import flashmla as fm

    num_blocks = (context + BLOCK_SIZE - 1) // BLOCK_SIZE
    nbytes = num_blocks * BLOCK_SIZE * MLA_ENTRY_BYTES
    buf, keep = alloc(nbytes, tier, device, numa_node)
    # dense fp8 layout: num_blocks x num_heads_k x (page_block_size * entry_bytes)
    cache = buf.view(num_blocks, 1, BLOCK_SIZE * MLA_ENTRY_BYTES)
    block_table = (
        torch.arange(num_blocks, dtype=torch.int32, device=device)
        .view(1, -1)
        .repeat(batch, 1)
    )
    cache_seqlens = torch.full(
        (batch,), context, dtype=torch.int32, device=device
    )
    q = torch.randn(
        (batch, q_tokens, MLA_HEADS, MLA_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    meta, splits = fm.get_mla_metadata_dense_fp8(cache_seqlens, q_tokens * MLA_HEADS, 1)

    def run():
        fm.flash_mla_with_kvcache_fp8(
            q=q,
            k_cache=cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=MLA_VALUE_DIM,
            tile_scheduler_metadata=meta,
            num_splits=splits,
            causal=False,
        )

    ms = timed(run, args.warmups, args.iterations)
    del keep
    # every sequence shares one cache here, so bytes touched is one context
    return ms, context * MLA_ENTRY_BYTES


def bench_gqa(
    context: int, q_tokens: int, batch: int, tier: str, device, numa_node, args
):
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    elt = 2  # bf16
    per_token = 2 * GQA_KV_HEADS * GQA_HEAD_DIM * elt
    total_k = context * batch
    nbytes = total_k * per_token
    buf, keep = alloc(nbytes, tier, device, numa_node)
    kv = buf.view(torch.bfloat16).view(2, total_k, GQA_KV_HEADS, GQA_HEAD_DIM)
    k, v = kv[0], kv[1]
    q = torch.randn(
        (q_tokens * batch, GQA_Q_HEADS, GQA_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    cu_q = torch.arange(
        0, q_tokens * (batch + 1), q_tokens, dtype=torch.int32, device=device
    )
    cu_k = torch.arange(
        0, context * (batch + 1), context, dtype=torch.int32, device=device
    )

    def run():
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_tokens,
            max_seqlen_k=context,
            softmax_scale=GQA_HEAD_DIM**-0.5,
            causal=False,
        )

    ms = timed(run, args.warmups, args.iterations)
    del keep
    return ms, nbytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numa-node", type=int, default=-1)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[4096, 32768, 131072]
    )
    parser.add_argument("--query-tokens", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[1],
        help="concurrent sequences; decode parallelism comes from here",
    )
    parser.add_argument("--kinds", nargs="+", default=["mla_dense", "gqa"])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    numa_node = args.numa_node
    if numa_node < 0:
        from vllm.platforms import current_platform

        numa_node = current_platform.get_device_numa_node(device.index or 0)
        if numa_node is None:
            raise SystemExit("could not determine the paired Grace NUMA node")
    print(f"Grace NUMA node {numa_node}, device {torch.cuda.get_device_name(device)}")

    benches = {"mla_dense": bench_mla_dense, "gqa": bench_gqa}
    rows = []
    print(
        f"\n{'kind':10} {'ctx':>8} {'tok':>4} {'bs':>4} {'HBM ms':>9} {'Grace ms':>9} "
        f"{'penalty':>8} {'HBM GB/s':>9} {'Grace GB/s':>11}"
    )
    for kind in args.kinds:
        for context in args.contexts:
            for tokens in args.query_tokens:
              for batch in args.batch:
                try:
                    res = {}
                    for tier in ("hbm", "host_uva"):
                        ms, nbytes = benches[kind](
                            context, tokens, batch, tier, device, numa_node, args
                        )
                        res[tier] = (ms, nbytes)
                    (h_ms, nbytes), (g_ms, _) = res["hbm"], res["host_uva"]
                    row = {
                        "kind": kind,
                        "context": context,
                        "query_tokens": tokens,
                        "batch": batch,
                        "kv_bytes_per_layer": nbytes,
                        "hbm_ms": h_ms,
                        "grace_ms": g_ms,
                        "penalty": g_ms / h_ms,
                        "hbm_gbps": nbytes / (h_ms / 1000) / 1e9,
                        "grace_gbps": nbytes / (g_ms / 1000) / 1e9,
                    }
                    rows.append(row)
                    print(
                        f"{kind:10} {context:>8} {tokens:>4} {batch:>4} "
                        f"{h_ms:9.3f} {g_ms:9.3f} "
                        f"{row['penalty']:7.2f}x {row['hbm_gbps']:9.1f} "
                        f"{row['grace_gbps']:11.1f}"
                    )
                except Exception as exc:  # noqa: BLE001 — sweep must survive
                    print(
                        f"{kind:10} {context:>8} {tokens:>4} {batch:>4}  "
                        f"FAILED: {exc}"[:150]
                    )
    print()
    print(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
