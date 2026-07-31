#!/usr/bin/env python3
"""Phase 1 gate: is one fused kernel per layer cheaper than today's alignment?

Times a whole rank-step - all 75 routed layers - as a captured CUDA graph, so
launch overhead is accounted the way it is in the served model.

Arms:

  baseline   today's path: two ``moe_align_block_size`` calls per layer against
             the static primary maps, no assignment. 4 device nodes per layer.
  fused      one ``fused_assign_align`` per layer: assignment plus both tiers'
             metadata. 1 device node per layer.

v1's greedy arm measured 375 nodes and 1.530 ms per rank-step at c4 (75
scheduler + 150 align + 150 sort). The Phase 1 gate is 75 nodes and 0.4 ms.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fused_assign_align import (
    EP,
    TierMaps,
    align_buffer_shapes,
    build_tier_maps,
    fused_assign_align,
    reference_assign,
)
from replay_exact import (
    NUM_LAYERS,
    load_placement,
    load_requests,
    route_counts,
    sample_steps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=(1, 4))
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--num-warps", type=int, nargs="+", default=(4,))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def time_graph(graph: torch.cuda.CUDAGraph, replays: int, warmup: int) -> float:
    """Mean milliseconds per replay."""
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / replays


def build_layer_inputs(
    steps: np.ndarray,
    counts: np.ndarray,
    owners: np.ndarray,
    hot: np.ndarray,
    secondary: np.ndarray,
    ep_rank: int,
    device: torch.device,
) -> list[dict]:
    layers = []
    for layer in range(NUM_LAYERS):
        hot_map, cold_map = build_tier_maps(
            owners[layer], hot[layer], secondary[layer], ep_rank
        )
        selected = reference_assign(
            counts[0, layer].astype(np.int64),
            owners[layer].astype(np.int64),
            secondary[layer].astype(np.int64),
            hot[layer].astype(np.int64),
        )
        active = counts[0, layer] > 0
        hot_mine = active & (hot[layer] != 0) & (owners[layer] == ep_rank)
        cold_mine = active & (hot[layer] == 0) & (selected == ep_rank)

        def to_device(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array)).to(device)

        layers.append(
            {
                "topk": to_device(steps[0, :, layer, :].astype(np.int32)),
                "primary": to_device(owners[layer].astype(np.int32)),
                "secondary": to_device(secondary[layer].astype(np.int32)),
                "hot": to_device(hot[layer].astype(np.int32)),
                "maps": TierMaps(to_device(hot_map), to_device(cold_map)),
                # The baseline aligns against the static primary maps, which is
                # exactly what the shipping path does today.
                "hot_static": to_device(np.where(hot_mine, hot_map, -1)),
                "cold_static": to_device(np.where(cold_mine, cold_map, -1)),
            }
        )
    return layers


def main() -> None:
    args = parse_args()
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    heldout = load_requests(args.trace_dir, "heldout")
    owners, hot, secondary = load_placement(args.placement)
    num_experts = owners.shape[1]

    results: dict = {
        "placement": str(args.placement),
        "device": torch.cuda.get_device_name(0),
        "replays": args.replays,
        "workloads": {},
    }

    for concurrency in args.concurrency:
        steps = sample_steps(heldout, 1, concurrency, rng)
        counts = route_counts(steps)
        layers = build_layer_inputs(
            steps, counts, owners, hot, secondary, args.ep_rank, device
        )
        num_routes = layers[0]["topk"].numel()

        # Preallocate every output so capture sees stable addresses.
        fused_by_warps = {}
        for warps in args.num_warps:
            outputs = [
                fused_assign_align(
                    entry["topk"], entry["primary"], entry["secondary"],
                    entry["hot"], entry["maps"], args.ep_rank,
                    args.block_m, args.block_m, num_warps=warps,
                )
                for entry in layers
            ]
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for entry, out in zip(layers, outputs):
                    fused_assign_align(
                        entry["topk"], entry["primary"], entry["secondary"],
                        entry["hot"], entry["maps"], args.ep_rank,
                        args.block_m, args.block_m, num_warps=warps, out=out,
                    )
            fused_by_warps[warps] = (graph, outputs)

        for entry in layers:
            for key in ("hot_static", "cold_static"):
                moe_align_block_size(
                    entry["topk"], args.block_m, num_experts, entry[key],
                    ignore_invalid_experts=True,
                )
        torch.cuda.synchronize()

        baseline_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(baseline_graph):
            for entry in layers:
                for key in ("hot_static", "cold_static"):
                    moe_align_block_size(
                        entry["topk"], args.block_m, num_experts, entry[key],
                        ignore_invalid_experts=True,
                    )

        by_warps = {
            warps: time_graph(graph, args.replays, args.warmup)
            for warps, (graph, _) in fused_by_warps.items()
        }
        best_warps = min(by_warps, key=by_warps.get)
        fused_ms = by_warps[best_warps]
        baseline_ms = time_graph(baseline_graph, args.replays, args.warmup)

        sorted_len, blocks = align_buffer_shapes(
            num_routes, num_experts, args.block_m
        )
        row = {
            "routes_per_layer": int(num_routes),
            "sorted_buffer": int(sorted_len),
            "block_buffer": int(blocks),
            "baseline_ms_per_rank_step": baseline_ms,
            "fused_ms_per_rank_step": fused_ms,
            "baseline_us_per_layer": 1000 * baseline_ms / NUM_LAYERS,
            "fused_us_per_layer": 1000 * fused_ms / NUM_LAYERS,
            "baseline_nodes": 4 * NUM_LAYERS,
            "fused_nodes": NUM_LAYERS,
            "delta_percent": 100 * (fused_ms - baseline_ms) / baseline_ms,
            # v1's greedy arm, from the paired c4 trace.
            "v1_greedy_ms_per_rank_step": 1.530,
            "v1_greedy_nodes": 375,
            "best_num_warps": best_warps,
            "ms_by_num_warps": {str(k): v for k, v in by_warps.items()},
        }
        results["workloads"][f"c{concurrency}"] = row
        print(
            f"c{concurrency}: {num_routes} routes/layer | "
            f"baseline {baseline_ms:.3f} ms ({4 * NUM_LAYERS} nodes) | "
            f"fused {fused_ms:.3f} ms ({NUM_LAYERS} nodes) | "
            f"{row['delta_percent']:+.1f}% vs baseline, "
            f"{100 * (fused_ms - 1.530) / 1.530:+.1f}% vs v1 greedy | "
            f"warps {best_warps} of "
            + str({k: round(v, 3) for k, v in by_warps.items()}),
            flush=True,
        )
        del fused_by_warps, baseline_graph, layers
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if args.output:
        args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
