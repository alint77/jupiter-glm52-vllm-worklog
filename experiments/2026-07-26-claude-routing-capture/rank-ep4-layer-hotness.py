#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EP_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def coverage_size(sorted_shares: np.ndarray, target: float) -> int:
    return int(np.searchsorted(np.cumsum(sorted_shares), target) + 1)


def local_metrics(layer_counts: np.ndarray) -> dict[str, float | int]:
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
    return {
        "effective_local_experts": float(np.exp(entropy)),
        "local_gini": float(gini),
        "top_8_local_route_share": float(sorted_shares[:8].sum()),
        "local_experts_for_50pct": coverage_size(sorted_shares, 0.5),
        "local_experts_for_80pct": coverage_size(sorted_shares, 0.8),
    }


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    counts = np.loadtxt(args.counts, delimiter=",", dtype=np.int64)
    profile = json.loads(args.profile.read_text())
    owners = np.asarray(profile["owners"], dtype=np.int16)
    layers = np.asarray(profile["routed_layers"], dtype=np.int16)
    if counts.shape != (75, 256) or owners.shape != counts.shape:
        raise ValueError("Counts and EP ownership must both have shape (75, 256)")
    for rank in range(EP_SIZE):
        if not np.all(np.sum(owners == rank, axis=1) == 64):
            raise ValueError(f"EP rank {rank} does not own 64 experts per layer")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_shares = np.zeros((EP_SIZE, len(layers)))
    rank_summaries = []
    for rank in range(EP_SIZE):
        records = []
        for layer_index, layer in enumerate(layers):
            local = counts[layer_index, owners[layer_index] == rank]
            route_count = int(local.sum())
            load_shares[rank, layer_index] = route_count / counts[layer_index].sum()
            hottest_local_index = int(np.argmax(local))
            owned_experts = np.flatnonzero(owners[layer_index] == rank)
            records.append(
                {
                    "layer": int(layer),
                    "route_count": route_count,
                    "layer_route_share": load_shares[rank, layer_index],
                    "load_vs_balanced": load_shares[rank, layer_index] * EP_SIZE,
                    "hottest_expert": int(owned_experts[hottest_local_index]),
                    "hottest_expert_rank_share": float(
                        local[hottest_local_index] / route_count
                    ),
                    **local_metrics(local),
                }
            )
        records.sort(key=lambda record: (-record["layer_route_share"], record["layer"]))
        ranked_records = [
            {"rank_layer_load_rank": index, **record}
            for index, record in enumerate(records, 1)
        ]
        write_csv(
            args.output_dir / f"ep-rank-{rank}-layer-hotness.csv",
            ranked_records,
        )
        rank_total = int(counts[owners == rank].sum())
        rank_summaries.append(
            {
                "ep_rank": rank,
                "total_route_count": rank_total,
                "total_route_share": rank_total / counts.sum(),
                "hottest_layers": [record["layer"] for record in records[:10]],
                "coldest_layers": [record["layer"] for record in records[-10:][::-1]],
                "maximum_layer_route_share": records[0]["layer_route_share"],
                "minimum_layer_route_share": records[-1]["layer_route_share"],
            }
        )

    critical_records = []
    for layer_index, layer in enumerate(layers):
        shares = load_shares[:, layer_index]
        critical_records.append(
            {
                "layer": int(layer),
                **{
                    f"rank_{rank}_route_share": float(shares[rank])
                    for rank in range(EP_SIZE)
                },
                "critical_rank": int(np.argmax(shares)),
                "maximum_rank_route_share": float(shares.max()),
                "minimum_rank_route_share": float(shares.min()),
                "rank_share_spread": float(shares.max() - shares.min()),
            }
        )
    critical_records.sort(
        key=lambda record: (-record["maximum_rank_route_share"], record["layer"])
    )
    write_csv(args.output_dir / "ep4-layer-criticality.csv", critical_records)

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")
    for rank, axis in enumerate(axes.flat):
        axis.plot(layers, 100 * load_shares[rank], color=colors[rank], linewidth=1.5)
        axis.axhline(25, color="#666666", linestyle="--", linewidth=1)
        axis.set_title(f"EP rank {rank}")
        axis.grid(alpha=0.2)
    figure.supxlabel("Model layer")
    figure.supylabel("Share of layer routes owned by rank (%)")
    figure.suptitle("EP4 per-layer routing load from 109-request Claude capture")
    figure.tight_layout()
    figure.savefig(args.output_dir / "ep4-layer-load.png", dpi=180)
    plt.close(figure)

    maximum_shares = load_shares.max(axis=0)
    summary = {
        "ownership_profile": str(args.profile),
        "balanced_layer_route_share": 0.25,
        "mean_critical_rank_route_share": float(maximum_shares.mean()),
        "p95_critical_rank_route_share": float(np.percentile(maximum_shares, 95)),
        "maximum_critical_rank_route_share": float(maximum_shares.max()),
        "layers_above_30pct_on_one_rank": int(np.sum(maximum_shares > 0.30)),
        "layers_above_35pct_on_one_rank": int(np.sum(maximum_shares > 0.35)),
        "layers_above_40pct_on_one_rank": int(np.sum(maximum_shares > 0.40)),
        "critical_layer_count_by_rank": np.bincount(
            np.argmax(load_shares, axis=0), minlength=EP_SIZE
        ).tolist(),
        "ranks": rank_summaries,
    }
    (args.output_dir / "ep4-layer-hotness-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
