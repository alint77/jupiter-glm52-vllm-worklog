#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def coverage_size(sorted_shares: np.ndarray, target: float) -> int:
    return int(np.searchsorted(np.cumsum(sorted_shares), target) + 1)


def main() -> None:
    args = parse_args()
    counts = np.loadtxt(args.counts, delimiter=",", dtype=np.int64)
    if counts.shape != (75, 256):
        raise ValueError(f"Expected a 75x256 grid, got {counts.shape}")

    records = []
    for layer_index, layer_counts in enumerate(counts):
        shares = layer_counts / layer_counts.sum()
        sorted_shares = np.sort(shares)[::-1]
        positive = shares > 0
        entropy = -np.sum(shares[positive] * np.log(shares[positive]))
        ordered_counts = np.sort(layer_counts)
        n = len(ordered_counts)
        gini = (
            2
            * np.sum(np.arange(1, n + 1) * ordered_counts)
            / (n * ordered_counts.sum())
            - (n + 1) / n
        )
        records.append(
            {
                "layer": layer_index + 3,
                "effective_experts": float(np.exp(entropy)),
                "gini": float(gini),
                "top_8_route_share": float(sorted_shares[:8].sum()),
                "top_16_route_share": float(sorted_shares[:16].sum()),
                "top_32_route_share": float(sorted_shares[:32].sum()),
                "experts_for_50pct": coverage_size(sorted_shares, 0.5),
                "experts_for_80pct": coverage_size(sorted_shares, 0.8),
                "hottest_expert": int(np.argmax(layer_counts)),
                "hottest_expert_share": float(sorted_shares[0]),
                "total_routes": int(layer_counts.sum()),
            }
        )

    records.sort(key=lambda record: (record["effective_experts"], record["layer"]))
    fieldnames = ["hotness_rank", *records[0]]
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(records, 1):
            writer.writerow({"hotness_rank": rank, **record})


if __name__ == "__main__":
    main()
