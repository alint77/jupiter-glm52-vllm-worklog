#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def load_hotness(path: Path) -> tuple[np.ndarray, np.ndarray]:
    counts = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if counts.shape != (75, 256):
        raise ValueError(f"{path} has shape {counts.shape}, expected (75, 256)")
    if (counts < 0).any() or (counts.sum(axis=1) == 0).any():
        raise ValueError(f"{path} contains invalid counts")
    relative = counts / (counts.sum(axis=1, keepdims=True) / counts.shape[1])
    return counts, relative.reshape(-1)


def summarize(counts: np.ndarray, relative: np.ndarray) -> dict[str, float | int]:
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    log_probabilities = np.zeros_like(probabilities)
    np.log(
        probabilities,
        out=log_probabilities,
        where=probabilities > 0,
    )
    entropy = -np.sum(probabilities * log_probabilities, axis=1)
    ordered = np.sort(relative)
    cell_count = len(ordered)
    gini = (
        2 * np.sum(np.arange(1, cell_count + 1) * ordered)
        / (cell_count * ordered.sum())
        - (cell_count + 1) / cell_count
    )
    top_count = int(np.ceil(cell_count * 0.01))
    return {
        "total_routes": int(counts.sum()),
        "zero_cells": int((counts == 0).sum()),
        "median_relative_hotness": float(np.median(relative)),
        "p95_relative_hotness": float(np.percentile(relative, 95)),
        "p99_relative_hotness": float(np.percentile(relative, 99)),
        "maximum_relative_hotness": float(relative.max()),
        "cells_above_4x_percent": float(100 * np.mean(relative > 4)),
        "top_1_percent_route_share_percent": float(
            100 * ordered[-top_count:].sum() / ordered.sum()
        ),
        "mean_effective_experts_per_layer": float(np.exp(entropy).mean()),
        "gini": float(gini),
    }


def main() -> None:
    args = parse_args()
    current_counts, current = load_hotness(args.current)
    previous_counts, previous = load_hotness(args.previous)

    labels = (
        "Claude Code · 108 valid natural requests",
        "Curated prompts · 24 × 256 output tokens",
    )
    colors = ("#00a6d6", "#e4572e")
    series = ((current, labels[0], colors[0]), (previous, labels[1], colors[1]))

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    linear_bins = np.linspace(0, 4, 81)
    for values, label, color in series:
        axes[0].hist(
            values,
            bins=linear_bins,
            weights=np.full(values.shape, 100 / values.size),
            histtype="step",
            linewidth=2,
            label=label,
            color=color,
        )
    axes[0].axvline(1, color="#666666", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Expert traffic / mean traffic in its layer",
        ylabel="Layer-expert cells per 0.05-wide bin (%)",
        title="Main distribution (tail above 4× omitted)",
        xlim=(0, 4),
    )
    axes[0].legend(frameon=False)

    positive = np.concatenate((current[current > 0], previous[previous > 0]))
    log_bins = np.geomspace(positive.min() * 0.9, positive.max() * 1.1, 70)
    for values, label, color in series:
        values = values[values > 0]
        axes[1].hist(
            values,
            bins=log_bins,
            weights=np.full(values.shape, 100 / values.size),
            histtype="step",
            linewidth=2,
            label=label,
            color=color,
        )
    axes[1].axvline(1, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Expert traffic / mean traffic in its layer (log scale)",
        ylabel="Layer-expert cells per logarithmic bin (%)",
        title="Full long tail (log–log)",
    )

    figure.suptitle(
        "GLM-5.2 expert-hotness distribution: real Claude use vs fixed prompts",
        fontsize=14,
    )
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    summary = {
        "normalization": "count / mean count within each routed layer",
        "current": summarize(current_counts, current),
        "previous": summarize(previous_counts, previous),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
