#!/usr/bin/env python3
"""Plot how skewed MiMo-V2.6's expert routing is and what that buys tiering.

Ranks come from the training split and every share is measured on the
held-out split (task families the ranking never saw), so the curves are what
a profile delivers on new traffic, not in-sample fit.

    plot_routing.py --trace-dir DIR --profile profile-3827.json --out figs/
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step_tiers import linear_even

STEP = 8
HBM = "#2a78d6"
DDR = "#eb6834"
MUTED = "#8a8984"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "legend.frameon": False,
    }
)


def load(trace_dir: Path, layers: list[int], num_experts: int):
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    counts = {s: np.zeros((len(layers), num_experts)) for s in ("train", "heldout")}
    steps = []
    for record in manifest:
        routes = np.load(trace_dir / record["file"])[:, layers, :].astype(np.int64)
        split = record["split"]
        for layer in range(len(layers)):
            counts[split][layer] += np.bincount(
                routes[:, layer].ravel(), minlength=num_experts
            )
        if split == "heldout":
            usable = routes.shape[0] // STEP * STEP
            steps.append(routes[:usable].reshape(-1, STEP, len(layers), 8))
    return counts["train"], counts["heldout"], np.concatenate(steps)


def step_presence(steps: np.ndarray, num_experts: int) -> np.ndarray:
    n_steps, _, n_layers, _ = steps.shape
    present = np.zeros((n_steps, n_layers, num_experts), dtype=bool)
    for layer in range(n_layers):
        ids = steps[:, :, layer, :].reshape(n_steps, -1)
        np.put_along_axis(present[:, layer, :], ids, True, axis=1)
    return present


def shares(hot: np.ndarray, heldout: np.ndarray, present: np.ndarray) -> dict:
    return {
        "routes": float((heldout * ~hot).sum() / heldout.sum()),
        "distinct": float((present & ~hot[None]).sum() / present.sum()),
        "distinct_per_layer": float(present.sum() / (present.shape[0] * present.shape[1])),
    }


def fig_skew(train: np.ndarray, out: Path) -> None:
    num_experts = train.shape[1]
    share = np.sort(train / train.sum(axis=1, keepdims=True), axis=1)[:, ::-1] * 100
    rank = np.arange(1, num_experts + 1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(
        rank, np.percentile(share, 10, axis=0), np.percentile(share, 90, axis=0),
        color=HBM, alpha=0.18, linewidth=0, label="10th-90th percentile of layers",
    )
    ax.plot(rank, np.median(share, axis=0), color=HBM, linewidth=2, label="median layer")
    ax.axhline(100 / num_experts, color=MUTED, linewidth=1.5, linestyle="--")
    ax.text(num_experts, 100 / num_experts * 1.12, "uniform (1/384 = 0.26%)",
            color=INK2, ha="right", va="bottom", fontsize=10)
    ax.axvline(num_experts / 2, color=INK2, linewidth=1, linestyle=":")
    ax.text(num_experts / 2, 0.97, "top half  |  bottom half ", color=INK2, fontsize=10,
            ha="center", va="top", transform=ax.get_xaxis_transform(),
            backgroundcolor=SURFACE)
    ax.set_yscale("log")
    ax.set_xlim(1, num_experts)
    ax.set_xlabel("expert rank within its layer (1 = most used)")
    ax.set_ylabel("share of the layer's routes (%)")
    top = np.median(share[:, 0]) / (100 / num_experts)
    top_half = np.median(share[:, : num_experts // 2].sum(axis=1))
    ax.set_title(f"Expert routing is skewed\nbusiest expert ~{top:.0f}x a uniform share; "
                 f"the top half takes {top_half:.0f}% of routes (in-sample)", pad=12, linespacing=1.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "1-expert-skew.png", dpi=160)
    plt.close(fig)


def fig_cumulative(train: np.ndarray, heldout: np.ndarray, actual_frac: float,
                   out: Path) -> dict:
    num_experts = train.shape[1]
    order = np.argsort(-train, axis=1, kind="stable")
    ranked = np.take_along_axis(heldout, order, axis=1)
    cum = np.cumsum(ranked, axis=1) / ranked.sum(axis=1, keepdims=True) * 100
    x = np.arange(1, num_experts + 1) / num_experts * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(x, np.percentile(cum, 10, axis=0), np.percentile(cum, 90, axis=0),
                    color=HBM, alpha=0.18, linewidth=0)
    ax.plot(x, np.median(cum, axis=0), color=HBM, linewidth=2,
            label="keep the most-used experts (ranked on other tasks)")
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1.5, linestyle="--",
            label="keep arbitrary experts (today's linear split)")
    marks = {}
    for frac, name in ((50.0, "top half"), (actual_frac * 100, "MiMo's HBM budget")):
        index = int(round(frac / 100 * num_experts)) - 1
        value = float(np.median(cum[:, index]))
        marks[name] = value
        ax.plot([frac], [value], "o", color=HBM, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.plot([frac, frac], [0, value], color=INK2, linewidth=0.8, linestyle=":")
        ax.annotate(f"{name}: {frac:.0f}% of experts in HBM\n-> {value:.0f}% of routes hit HBM, "
                    f"{100 - value:.0f}% go to DDR",
                    (frac, value), xytext=(12, -38 if name == "top half" else -8),
                    textcoords="offset points", fontsize=10, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("experts kept in HBM, per layer (%)")
    ax.set_ylabel("held-out routes served from HBM (%)")
    ax.set_title("Keeping the popular half in HBM serves most routes", pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "2-cumulative-hbm-share.png", dpi=160)
    plt.close(fig)
    return marks


def fig_tiers(results: dict, out: Path) -> None:
    labels = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, key, title in (
        (axes[0], "routes", "Share of routed activations"),
        (axes[1], "distinct", "Share of experts read per 8-token decode step"),
    ):
        y = np.arange(len(labels))[::-1]
        ddr = np.array([results[label][key] * 100 for label in labels])
        hbm = 100 - ddr
        ax.barh(y, hbm, color=HBM, height=0.55, edgecolor=SURFACE, linewidth=2,
                label="HBM")
        ax.barh(y, ddr, left=hbm, color=DDR, height=0.55, edgecolor=SURFACE, linewidth=2,
                label="DDR (Grace, over C2C)")
        for yi, h, d in zip(y, hbm, ddr):
            ax.text(h / 2, yi, f"{h:.0f}%", ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")
            ax.text(h + d / 2, yi, f"{d:.0f}%", ha="center", va="center", color=INK,
                    fontsize=10, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("%")
    axes[0].set_yticks(np.arange(len(labels))[::-1], labels)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.suptitle("Where expert reads come from, on held-out traffic",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / "3-hbm-vs-ddr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_per_layer(train: np.ndarray, heldout: np.ndarray, layers: list[int],
                  out: Path) -> None:
    half = train.shape[1] // 2
    order = np.argsort(-train, axis=1, kind="stable")
    ranked = np.take_along_axis(heldout, order, axis=1)
    ddr = ranked[:, half:].sum(axis=1) / ranked.sum(axis=1) * 100
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(layers, ddr, color=DDR, width=0.8, edgecolor=SURFACE, linewidth=1)
    ax.axhline(50, color=MUTED, linestyle="--", linewidth=1.5)
    ax.text(layers[-1], 51, "arbitrary half offloaded: 50%", ha="right", va="bottom",
            color=INK2, fontsize=10)
    ax.set_xlim(layers[0] - 1, layers[-1] + 1)
    ax.set_ylim(0, 60)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("routes that land in DDR (%)")
    ax.set_title(f"Bottom half offloaded, per layer: {ddr.min():.0f}-{ddr.max():.0f}% of "
                 f"routes go to DDR (median {np.median(ddr):.0f}%)", pad=12)
    fig.tight_layout()
    fig.savefig(out / "4-per-layer-ddr.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    profile = json.loads(args.profile.read_text())
    layers = profile["routed_layers"]
    num_experts = profile["num_experts"]
    train, heldout, steps = load(args.trace_dir, layers, num_experts)
    present = step_presence(steps, num_experts)

    slots = sum(len(ids) for ids in profile["hot_experts"]) // 4
    actual_frac = slots / (len(layers) * num_experts // 4)
    _, linear_hot = linear_even(num_experts, len(layers), slots)
    top_half = np.zeros_like(train, dtype=bool)
    np.put_along_axis(top_half, np.argsort(-train, axis=1)[:, : num_experts // 2], True, axis=1)
    profile_hot = np.zeros_like(train, dtype=bool)
    for layer, ids in enumerate(profile["hot_experts"]):
        profile_hot[layer, ids] = True
    results = {
        f"linear, {actual_frac:.0%} in HBM (today)": shares(linear_hot, heldout, present),
        "top half in HBM (50%)": shares(top_half, heldout, present),
        f"profile, {actual_frac:.0%} in HBM": shares(profile_hot, heldout, present),
    }

    fig_skew(train, args.out)
    marks = fig_cumulative(train, heldout, actual_frac, args.out)
    fig_tiers(results, args.out)
    fig_per_layer(train, heldout, layers, args.out)
    summary = {"hot_fraction": actual_frac, "cumulative_marks": marks, "tiers": results}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
