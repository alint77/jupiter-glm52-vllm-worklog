#!/usr/bin/env python3

import argparse
import json
import math
import statistics

import torch

from vllm.model_executor.offloader.grace import GraceAllocation
from vllm.v1.attention.ops import flashmla as fm

LAYERS = 78
NUM_BLOCKS = 6251
BLOCK_SIZE = 64
ENTRY_BYTES = 656
TOPK = 2048
HEADS = 64
HEAD_DIM = 576
VALUE_DIM = 512


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def make_index_sets(pattern: str, device: torch.device) -> list[torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(17)
    sets = []
    for index in range(21):
        if pattern == "random":
            values = torch.randperm(400_000, generator=generator, device=device)[:TOPK]
        elif pattern == "sorted":
            values = torch.randperm(400_000, generator=generator, device=device)[:TOPK]
            values = values.sort().values
        else:
            start = (index * 19_003) % (400_000 - TOPK)
            values = torch.arange(start, start + TOPK, device=device)
        sets.append(values.to(torch.int32).view(1, 1, TOPK))
    return sets


def layer_index_set(index_sets: list[torch.Tensor], layer: int) -> torch.Tensor:
    if layer < 2:
        return index_sets[layer]
    return index_sets[2 + (layer - 2) // 4]


def run_token(
    caches: list[torch.Tensor],
    q: torch.Tensor,
    index_sets: list[torch.Tensor],
    metadata: fm.FlashMLASchedMeta,
    output: torch.Tensor,
) -> torch.Tensor:
    for layer, cache in enumerate(caches):
        output, _ = fm.flash_mla_with_kvcache(
            q=q,
            k_cache=cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=VALUE_DIM,
            tile_scheduler_metadata=metadata,
            is_fp8_kvcache=True,
            indices=layer_index_set(index_sets, layer),
            softmax_scale=HEAD_DIM**-0.5,
            out=output,
        )
    return output


def measure(
    caches: list[torch.Tensor],
    q: torch.Tensor,
    index_sets: list[torch.Tensor],
    warmups: int,
    iterations: int,
    use_cuda_graph: bool,
) -> tuple[dict[str, float], torch.Tensor]:
    metadata, _ = fm.get_mla_metadata()
    output = torch.empty((1, 1, HEADS, VALUE_DIM), dtype=q.dtype, device=q.device)
    for _ in range(warmups):
        output = run_token(caches, q, index_sets, metadata, output)
    torch.cuda.synchronize()

    graph = None
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run_token(caches, q, index_sets, metadata, output)
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph is None:
            output = run_token(caches, q, index_sets, metadata, output)
        else:
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    bytes_per_token = LAYERS * TOPK * ENTRY_BYTES
    median_ms = statistics.median(times)
    return (
        {
            "median_ms": median_ms,
            "p95_ms": percentile(times, 0.95),
            "p99_ms": percentile(times, 0.99),
            "min_ms": min(times),
            "max_ms": max(times),
            "effective_gbps_median": bytes_per_token / (median_ms / 1000) / 1e9,
        },
        output.clone(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        raise RuntimeError(reason)

    torch.manual_seed(17)
    device = torch.device("cuda:0")
    q = torch.randn((1, 1, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    cache_shape = (NUM_BLOCKS, BLOCK_SIZE, 1, ENTRY_BYTES)

    grace_allocations = []
    grace_caches = []
    for _ in range(LAYERS):
        allocation = GraceAllocation.allocate_pinned(
            (NUM_BLOCKS * BLOCK_SIZE * ENTRY_BYTES,),
            torch.int8,
            device.index or 0,
            args.numa_node,
        )
        allocation.cpu_tensor.zero_()
        grace_allocations.append(allocation)
        grace_caches.append(allocation.cuda_alias.view(torch.uint8).view(cache_shape))

    hbm_caches = [
        torch.zeros(cache_shape, dtype=torch.uint8, device=device)
        for _ in range(LAYERS)
    ]
    locality_before = [
        allocation.audit_numa(samples=4) for allocation in grace_allocations
    ]

    results = {}
    correctness = {}
    for pattern in ("random", "sorted", "clustered"):
        index_sets = make_index_sets(pattern, device)
        results[pattern] = {
            "eager": {},
            "cuda_graph": {},
        }
        outputs = {}
        for mode, use_cuda_graph in (("eager", False), ("cuda_graph", True)):
            for tier, caches in (("host_uva", grace_caches), ("hbm", hbm_caches)):
                result, output = measure(
                    caches,
                    q,
                    index_sets,
                    args.warmups,
                    args.iterations,
                    use_cuda_graph,
                )
                results[pattern][mode][tier] = result
                outputs[(mode, tier)] = output
            torch.testing.assert_close(
                outputs[(mode, "host_uva")],
                outputs[(mode, "hbm")],
                rtol=0,
                atol=0,
            )
        correctness[pattern] = "exact across tiers"

    locality_after = [
        allocation.audit_numa(samples=4) for allocation in grace_allocations
    ]
    report = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "layers": LAYERS,
        "num_blocks": NUM_BLOCKS,
        "block_size": BLOCK_SIZE,
        "entry_bytes": ENTRY_BYTES,
        "topk": TOPK,
        "working_set_bytes": LAYERS * NUM_BLOCKS * BLOCK_SIZE * ENTRY_BYTES,
        "sparse_read_bytes_per_token": LAYERS * TOPK * ENTRY_BYTES,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "results": results,
        "correctness": correctness,
        "minimum_local_fraction_before": min(
            placement.local_fraction for placement in locality_before
        ),
        "minimum_local_fraction_after": min(
            placement.local_fraction for placement in locality_after
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
