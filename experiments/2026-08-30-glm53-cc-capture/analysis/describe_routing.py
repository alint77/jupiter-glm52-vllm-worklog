#!/usr/bin/env python3
"""Descriptive analysis of a routing capture, plus scoring of shipped profiles.

Answers three questions about a live Claude Code capture: what the corpus looks
like, how concentrated GLM-5.3's router actually is under real traffic, and what
the shipped placement profile costs on it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from agent_space.benchmarks.optimize_routing_profile import (
    NUM_EXPERTS,
    ROUTED_LAYERS,
    evaluate,
    load_requests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def profile_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    if data["routed_layers"] != list(ROUTED_LAYERS):
        raise ValueError(f"{path}: routed layers do not match")
    owners = np.asarray(data["owners"], dtype=np.int16)
    hot = np.zeros((len(ROUTED_LAYERS), NUM_EXPERTS), dtype=bool)
    for layer, experts in enumerate(data["hot_experts"]):
        hot[layer, np.asarray(experts, dtype=np.int64)] = True
    return owners, hot


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.trace_dir / "manifest.json").read_text())
    split = {r["request_hash"]: r.get("split", "train") for r in manifest}
    requests = load_requests(args.trace_dir)

    counts = np.zeros((len(ROUTED_LAYERS), NUM_EXPERTS), dtype=np.int64)
    lengths = []
    for _, routes in requests:
        lengths.append(routes.shape[0])
        for layer in range(len(ROUTED_LAYERS)):
            counts[layer] += np.bincount(
                routes[:, layer, :].ravel(), minlength=NUM_EXPERTS
            )
    lengths = np.asarray(lengths)
    total = counts.sum()

    report: dict = {
        "corpus": {
            "requests": len(requests),
            "routed_positions": int(lengths.sum()),
            "expert_activations": int(total),
            "positions_per_request": {
                "min": int(lengths.min()),
                "median": float(np.median(lengths)),
                "mean": float(lengths.mean()),
                "p90": float(np.percentile(lengths, 90)),
                "max": int(lengths.max()),
            },
        }
    }

    # Concentration. Sorted mass per layer says how much of the routing a given
    # residency budget can capture at best.
    ordered = -np.sort(-counts, axis=1)
    cumulative = ordered.cumsum(axis=1) / counts.sum(axis=1, keepdims=True)
    report["concentration"] = {
        f"top_{k}_share": {
            "mean": float(cumulative[:, k - 1].mean()),
            "min_layer": float(cumulative[:, k - 1].min()),
            "max_layer": float(cumulative[:, k - 1].max()),
        }
        for k in (16, 32, 64, 128, 160, 192)
    }
    # Gini per layer, as a scale-free skew measure.
    gini = []
    for layer in range(len(ROUTED_LAYERS)):
        x = np.sort(counts[layer]).astype(np.float64)
        n = x.size
        gini.append(float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())))
    gini = np.asarray(gini)
    report["gini"] = {
        "mean": float(gini.mean()),
        "min": float(gini.min()),
        "argmin_layer": int(ROUTED_LAYERS[int(gini.argmin())]),
        "max": float(gini.max()),
        "argmax_layer": int(ROUTED_LAYERS[int(gini.argmax())]),
    }
    never = (counts == 0).sum(axis=1)
    report["dead_experts"] = {
        "layers_with_any": int((never > 0).sum()),
        "mean_per_layer": float(never.mean()),
        "max_per_layer": int(never.max()),
    }

    heldout = [r for r in requests if split.get(r[0]) == "heldout"]
    train = [r for r in requests if split.get(r[0]) != "heldout"]
    report["split"] = {"train": len(train), "heldout": len(heldout)}

    # Empirical ceiling: rank by train-split frequency at the shipped budget.
    scored = {}
    for spec in args.profile:
        label, _, path = spec.partition("=")
        owners, hot = profile_arrays(Path(path))
        budget = int(hot.sum())
        scored[label] = {
            "hot_experts": budget,
            "heldout": evaluate(heldout, owners, hot)["routing_cold_hit_rate"],
            "train": evaluate(train, owners, hot)["routing_cold_hit_rate"],
        }
        # Oracle at the same per-layer budget, ranked on the train split only.
        per_layer = hot.sum(axis=1)
        train_counts = np.zeros_like(counts)
        for _, routes in train:
            for layer in range(len(ROUTED_LAYERS)):
                train_counts[layer] += np.bincount(
                    routes[:, layer, :].ravel(), minlength=NUM_EXPERTS
                )
        oracle = np.zeros_like(hot)
        for layer in range(len(ROUTED_LAYERS)):
            pick = np.argsort(-train_counts[layer])[: per_layer[layer]]
            oracle[layer, pick] = True
        scored[label]["frequency_reranked_heldout"] = evaluate(
            heldout, owners, oracle
        )["routing_cold_hit_rate"]
        scored[label]["hot_set_overlap"] = float(
            (hot & oracle).sum() / max(int(hot.sum()), 1)
        )
        even = np.zeros_like(hot)
        for layer in range(len(ROUTED_LAYERS)):
            even[layer, : per_layer[layer]] = True
        scored[label]["even_placement_heldout"] = evaluate(heldout, owners, even)[
            "routing_cold_hit_rate"
        ]
    report["profiles"] = scored

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
