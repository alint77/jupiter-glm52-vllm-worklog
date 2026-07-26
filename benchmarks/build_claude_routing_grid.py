#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROUTED_LAYERS = tuple(range(3, 78))
NUM_EXPERTS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-limit", type=int)
    return parser.parse_args()


def is_default_route_trace(routes: np.ndarray) -> bool:
    default = np.arange(NUM_EXPERTS)[:8]
    return all(
        np.all(routes[:, model_layer, :] == default)
        for model_layer in ROUTED_LAYERS
    )


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in (args.trace_dir / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.record_limit is not None:
        records = records[: args.record_limit]
    if not records:
        raise ValueError("No Claude routing traces were recorded")

    counts = np.zeros((len(ROUTED_LAYERS), NUM_EXPERTS), dtype=np.int64)
    valid_records = []
    excluded_files = []
    for record in records:
        routes = np.load(args.trace_dir / record["file"], mmap_mode="r")
        if routes.ndim != 3 or routes.shape[1:] != (78, 8):
            raise ValueError(f"Invalid route shape {routes.shape}")
        if is_default_route_trace(routes):
            excluded_files.append(record["file"])
            continue
        valid_records.append(record)
        for layer, model_layer in enumerate(ROUTED_LAYERS):
            counts[layer] += np.bincount(
                routes[:, model_layer, :].reshape(-1), minlength=NUM_EXPERTS
            )
    if not valid_records:
        raise ValueError("Every Claude routing trace contained default routes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / "expert-hotness-counts.csv",
        counts,
        fmt="%d",
        delimiter=",",
    )

    layer_totals = counts.sum(axis=1)
    ranking = sorted(
        (
            (int(counts[layer, expert]), layer, expert)
            for layer in range(len(ROUTED_LAYERS))
            for expert in range(NUM_EXPERTS)
        ),
        reverse=True,
    )
    with (args.output_dir / "expert-hotness-ranking.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("rank", "layer", "expert", "count", "layer_fraction"))
        for rank, (count, layer, expert) in enumerate(ranking, 1):
            writer.writerow(
                (
                    rank,
                    ROUTED_LAYERS[layer],
                    expert,
                    count,
                    count / layer_totals[layer],
                )
            )

    masked = np.ma.masked_equal(counts, 0)
    figure, axis = plt.subplots(figsize=(20, 7))
    image = axis.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        norm=LogNorm(vmin=1, vmax=int(counts.max())),
        cmap="magma",
    )
    axis.set_xlabel("Expert ID")
    axis.set_ylabel("Routed layer")
    axis.set_yticks(range(0, len(ROUTED_LAYERS), 5))
    axis.set_yticklabels(ROUTED_LAYERS[::5])
    figure.colorbar(image, ax=axis, label="Target-router selections")
    figure.tight_layout()
    figure.savefig(args.output_dir / "expert-hotness-heatmap.png", dpi=180)
    plt.close(figure)

    summary = {
        "manifest_record_count": len(records),
        "request_count": len(valid_records),
        "excluded_default_route_files": excluded_files,
        "routed_positions": sum(r["routed_positions"] for r in valid_records),
        "total_routes": int(counts.sum()),
        "maximum_cell_count": int(counts.max()),
        "zero_cells": int((counts == 0).sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
