#!/usr/bin/env python3

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
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


def make_index_sets(
    pattern: str,
    device: torch.device,
    real_index_trace: Path | None = None,
    query_tokens: int = 1,
    index_mode: str = "shared",
) -> list[torch.Tensor]:
    """Top-k index sets shaped (1, query_tokens, TOPK).

    `index_mode` decides what the extra query tokens select. MTP3's draft
    positions are consecutive and attend to almost the same context, so their
    top-k sets overlap heavily -- `shared` models that and is the realistic
    case. `independent` gives every token its own draw, which is the worst case
    for cache reuse and an upper bound on traffic.
    """
    if pattern == "real":
        if real_index_trace is None:
            raise ValueError("The real pattern requires --real-index-trace")
        with np.load(real_index_trace) as data:
            source = data["indices"].astype(np.int64)
            context_position = int(data["context_position"])
        if (
            source.shape != (21, TOPK)
            or source.min() < 0
            or source.max() > context_position
            or (np.diff(np.sort(source, axis=1), axis=1) == 0).any()
        ):
            raise ValueError("Real DSA trace must contain 21 complete top-2048 sets")
        scale = (NUM_BLOCKS * BLOCK_SIZE - 1) / context_position
        scaled = np.floor(source * scale).astype(np.int32)
        return [
            torch.from_numpy(row).to(device).view(1, 1, TOPK).expand(1, query_tokens, TOPK).contiguous()
            for row in scaled
        ]

    generator = torch.Generator(device=device).manual_seed(17)
    sets = []
    for index in range(21):
        rows = []
        draws = query_tokens if index_mode == "independent" else 1
        for draw in range(draws):
            if pattern == "random":
                values = torch.randperm(400_000, generator=generator, device=device)[
                    :TOPK
                ]
            elif pattern == "sorted":
                values = torch.randperm(400_000, generator=generator, device=device)[
                    :TOPK
                ]
                values = values.sort().values
            else:
                start = ((index + draw) * 19_003) % (400_000 - TOPK)
                values = torch.arange(start, start + TOPK, device=device)
            rows.append(values.to(torch.int32))
        stacked = torch.stack(rows, dim=0)
        if draws == 1 and query_tokens > 1:
            stacked = stacked.expand(query_tokens, TOPK).contiguous()
        sets.append(stacked.view(1, query_tokens, TOPK))
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
    output = torch.empty(
        (1, q.shape[1], HEADS, VALUE_DIM), dtype=q.dtype, device=q.device
    )
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

    query_tokens = q.shape[1]
    bytes_per_token = LAYERS * TOPK * ENTRY_BYTES * query_tokens
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
    parser.add_argument(
        "--numa-node",
        type=int,
        default=-1,
        help="paired Grace NUMA node; -1 auto-detects (it is node-specific)",
    )
    parser.add_argument(
        "--query-tokens",
        type=int,
        nargs="+",
        default=[1],
        help="query tokens per step to sweep; MTP3 is 4 at c1 and 16 at c4",
    )
    parser.add_argument(
        "--index-mode",
        choices=("shared", "independent"),
        default="shared",
        help="whether extra query tokens reuse one top-k set or draw their own",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--real-index-trace", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        raise RuntimeError(reason)

    torch.manual_seed(17)
    device = torch.device("cuda:0")
    cache_shape = (NUM_BLOCKS, BLOCK_SIZE, 1, ENTRY_BYTES)

    numa_node = args.numa_node
    if numa_node < 0:
        from vllm.platforms import current_platform

        numa_node = current_platform.get_device_numa_node(device.index or 0)
        if numa_node is None:
            raise SystemExit("could not determine the paired Grace NUMA node")
        print(f"detected Grace NUMA node {numa_node} for GPU {device.index or 0}")

    grace_allocations = []
    grace_caches = []
    for _ in range(LAYERS):
        allocation = GraceAllocation.allocate_pinned(
            (NUM_BLOCKS * BLOCK_SIZE * ENTRY_BYTES,),
            torch.int8,
            device.index or 0,
            numa_node,
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
    patterns = ["random", "sorted", "clustered"]
    if args.real_index_trace is not None:
        patterns.append("real")
    print(
        f"{'pattern':10} {'tok':>4} {'mode':11} {'HBM ms':>9} {'Grace ms':>9} "
        f"{'penalty':>8} {'HBM GB/s':>9} {'Grace GB/s':>11}"
    )
    for tokens in args.query_tokens:
        q = torch.randn(
            (1, tokens, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        for pattern in patterns:
            index_sets = make_index_sets(
                pattern,
                device,
                args.real_index_trace,
                query_tokens=tokens,
                index_mode=args.index_mode,
            )
            key = f"{pattern}/t{tokens}"
            results[key] = {"eager": {}, "cuda_graph": {}}
            outputs = {}
            for mode, use_cuda_graph in (("eager", False), ("cuda_graph", True)):
                for tier, caches in (
                    ("host_uva", grace_caches),
                    ("hbm", hbm_caches),
                ):
                    result, output = measure(
                        caches,
                        q,
                        index_sets,
                        args.warmups,
                        args.iterations,
                        use_cuda_graph,
                    )
                    results[key][mode][tier] = result
                    outputs[(mode, tier)] = output
                torch.testing.assert_close(
                    outputs[(mode, "host_uva")],
                    outputs[(mode, "hbm")],
                    rtol=0,
                    atol=0,
                )
                hbm = results[key][mode]["hbm"]
                host = results[key][mode]["host_uva"]
                print(
                    f"{pattern:10} {tokens:>4} {mode:11} "
                    f"{hbm['median_ms']:9.3f} {host['median_ms']:9.3f} "
                    f"{host['median_ms'] / hbm['median_ms']:7.2f}x "
                    f"{hbm['effective_gbps_median']:9.1f} "
                    f"{host['effective_gbps_median']:11.1f}"
                )
            correctness[key] = "exact across tiers"

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
        "real_index_trace": str(args.real_index_trace)
        if args.real_index_trace is not None
        else None,
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
