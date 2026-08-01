#!/usr/bin/env python3
"""Fit per-tier cost against active expert count, from trace plus replay.

To decide how many HBM slots each layer should get, we need to know what a
slot buys. The existing calibration measured the Grace slope well (~46 us per
distinct active cold expert) but could not resolve the HBM slope, which sat
below the benchmark's fixed per-call overhead.

Here the two are recovered in situ. The trace gives, per routed layer, the mean
wall time of the hot and cold Marlin chains. The replay gives, for the same
layer under the same placement, the mean number of active hot and cold experts.
Regressing one on the other across the 75 layers yields a slope and intercept
per tier, at the shapes the model actually runs.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

import numpy as np

from analyze_marlin_overlap import (
    ALLREDUCE,
    GPU_CATEGORIES,
    MARLIN,
    RUNTIME_CATEGORIES,
    STEP_MARKER,
    TOPK_ANCHOR,
    load,
)


def _is_hot(event: dict) -> bool:
    """Hot tier launches 2 CTAs per SM, cold tier 1 (see the launch policy)."""
    return float(event["args"].get("blocks per SM", 0)) >= 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-graph-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def observed_layer_times(directory: Path, max_steps: int) -> dict[int, dict]:
    """Mean hot and cold chain microseconds for each routed layer."""
    hot: dict[int, list[float]] = collections.defaultdict(list)
    cold: dict[int, list[float]] = collections.defaultdict(list)
    for path in sorted(Path(directory).glob("*.pt.trace.json.gz")):
        events = load(path)
        by_correlation: dict[int, list[dict]] = collections.defaultdict(list)
        for event in events:
            correlation = event.get("args", {}).get("correlation")
            if event.get("cat") in GPU_CATEGORIES and correlation is not None:
                by_correlation[correlation].append(event)
        runtime = [e for e in events if e.get("cat") in RUNTIME_CATEGORIES]
        annotations = sorted(
            (
                e
                for e in events
                if e.get("cat") == "user_annotation" and STEP_MARKER in e["name"]
            ),
            key=lambda e: e["ts"],
        )
        for annotation in annotations[:max_steps]:
            end = annotation["ts"] + annotation["dur"]
            launches = [
                e
                for e in runtime
                if annotation["ts"] <= e["ts"] < end and "GraphLaunch" in e["name"]
            ]
            if len(launches) != 1:
                continue
            ops = by_correlation[launches[0]["args"]["correlation"]]
            anchors = sorted(
                (e for e in ops if TOPK_ANCHOR in e["name"]), key=lambda e: e["ts"]
            )
            graph_end = max(float(e["ts"] + e["dur"]) for e in ops)
            for index, anchor in enumerate(anchors):
                stop = (
                    float(anchors[index + 1]["ts"])
                    if index + 1 < len(anchors)
                    else graph_end
                )
                segment = [
                    e
                    for e in ops
                    if float(anchor["ts"]) <= float(e["ts"]) < stop
                    and MARLIN in e["name"]
                ]
                if len(segment) != 4:
                    continue
                # The tiers are told apart by grid, not by stream: the
                # launch policy gives the hot tier 2 CTAs per SM and the cold
                # tier 1, while CUDA graph replay rotates stream ids from one
                # layer to the next so stream identity means nothing here.
                h = sum(
                    float(e["dur"]) for e in segment if _is_hot(e)
                )
                c = sum(
                    float(e["dur"]) for e in segment if not _is_hot(e)
                )
                if h and c:
                    hot[index].append(h)
                    cold[index].append(c)
    return {
        layer: {
            "hot_us": statistics.mean(hot[layer]),
            "cold_us": statistics.mean(cold[layer]),
            "samples": len(hot[layer]),
        }
        for layer in sorted(hot)
    }


def expected_active(
    routes: Path, placement: Path, steps: int, concurrency: int, seed: int
) -> dict[int, dict]:
    """Mean active hot and cold expert counts per layer, per rank."""
    import sys

    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent.parent / "2026-07-31-replica-scheduling-v2"),
    )
    from fused_assign_align import reference_assign
    from replay_exact import (
        EP,
        NUM_LAYERS,
        load_placement,
        load_requests,
        route_counts,
        sample_steps,
    )

    rng = np.random.default_rng(seed)
    heldout = load_requests(routes, "heldout")
    owners, hot, secondary = load_placement(placement)
    counts = route_counts(sample_steps(heldout, steps, concurrency, rng))

    hot_active = np.zeros(NUM_LAYERS)
    cold_active = np.zeros(NUM_LAYERS)
    for step in range(counts.shape[0]):
        for layer in range(NUM_LAYERS):
            column = counts[step, layer]
            selected = reference_assign(
                column.astype(np.int64),
                owners[layer].astype(np.int64),
                secondary[layer].astype(np.int64),
                hot[layer].astype(np.int64),
            )
            active = column > 0
            # Per-rank means: the tiers run per rank, so divide by EP.
            hot_active[layer] += np.count_nonzero(active & (hot[layer] != 0)) / EP
            cold_active[layer] += np.count_nonzero(active & (hot[layer] == 0)) / EP
            del selected
    n = counts.shape[0]
    return {
        layer: {
            "hot_experts": hot_active[layer] / n,
            "cold_experts": cold_active[layer] / n,
        }
        for layer in range(NUM_LAYERS)
    }


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares slope, intercept and R^2."""
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(slope), float(intercept), 1.0 - ss_res / ss_tot


def main() -> None:
    args = parse_args()
    observed = observed_layer_times(args.trace_dir, args.max_graph_steps)
    predicted = expected_active(
        args.routes, args.placement, args.steps, args.concurrency, args.seed
    )
    layers = sorted(set(observed) & set(predicted))
    if not layers:
        raise SystemExit("no layers in common between trace and replay")

    hot_n = np.array([predicted[i]["hot_experts"] for i in layers])
    cold_n = np.array([predicted[i]["cold_experts"] for i in layers])
    hot_t = np.array([observed[i]["hot_us"] for i in layers])
    cold_t = np.array([observed[i]["cold_us"] for i in layers])

    hot_slope, hot_base, hot_r2 = fit(hot_n, hot_t)
    cold_slope, cold_base, cold_r2 = fit(cold_n, cold_t)

    result = {
        "layers": len(layers),
        "hot": {
            "us_per_expert": hot_slope,
            "base_us": hot_base,
            "r2": hot_r2,
            "mean_experts": float(hot_n.mean()),
            "mean_us": float(hot_t.mean()),
        },
        "cold": {
            "us_per_expert": cold_slope,
            "base_us": cold_base,
            "r2": cold_r2,
            "mean_experts": float(cold_n.mean()),
            "mean_us": float(cold_t.mean()),
        },
    }
    for tier in ("hot", "cold"):
        row = result[tier]
        print(
            f"{tier:5s}: {row['us_per_expert']:7.3f} us/expert + "
            f"{row['base_us']:7.2f} us base   R2={row['r2']:.3f}   "
            f"mean {row['mean_experts']:5.2f} experts -> {row['mean_us']:6.1f} us"
        )
    print(
        f"\ncold costs {cold_slope / hot_slope:.2f}x an HBM expert; "
        f"moving one expert from Grace to HBM shifts "
        f"{cold_slope:.1f} us off the cold chain and adds {hot_slope:.1f} us "
        f"to the hot chain."
    )

    if args.output:
        args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
