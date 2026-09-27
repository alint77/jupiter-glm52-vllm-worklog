#!/usr/bin/env python3
"""Tiered kernel vs production Marlin at the pinned sustained clock.

    plot_vs_marlin.py fixedclock-v71.json --out figs/vs-marlin.png

Top: layer MoE time (w13 + w2) against hot experts, one panel per cold count.
Bottom left: speedup over the (hot, cold) grid, trace-probable cells outlined.
Bottom right: trace-weighted layer time and per-step saving.
Marlin is max(hot tier, cold tier), i.e. perfect two-stream overlap: its best case.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-routing-profile"))
from plot_routing import GRID, HBM, INK, INK2, MUTED, SURFACE  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_fixedclock import cells  # noqa: E402

MARLIN = "#9a9993"


def load(path: Path):
    d = json.loads(path.read_text())
    hot = {int(k): v["total"] for k, v in d["marlin_hot"].items()}
    cold = {int(k): v["total"] for k, v in d["marlin_cold"].items()}
    tier = {}
    for k, v in d["tiered"].items():
        g, h, c = k.split("-")
        tier.setdefault((int(h), int(c)), {})[g] = v
    tiered = {hc: v["w13"] + v["w2"] for hc, v in tier.items() if len(v) == 2}

    def marlin(h, c):
        mh = np.interp(h, sorted(hot), [hot[k] for k in sorted(hot)]) if h else 0.0
        mc = np.interp(c, sorted(cold), [cold[k] for k in sorted(cold)]) if c else 0.0
        return max(mh, mc)

    return marlin, tiered, d["cells"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    marlin, tiered, weighted = load(args.json)
    prob = cells()

    fig = plt.figure(figsize=(13, 9.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.42, wspace=0.28)

    # --- small multiples: time vs hot experts, per cold count
    for ci, c in enumerate(range(4)):
        ax = fig.add_subplot(gs[0, ci])
        hs = sorted(h for (h, cc) in tiered if cc == c and (h > 0 or c > 0))
        mt = [marlin(h, c) for h in hs]
        tt = [tiered[(h, c)] for h in hs]
        ax.fill_between(hs, tt, mt, color=HBM, alpha=0.10, linewidth=0)
        ax.plot(hs, mt, "o--", color=MARLIN, linewidth=2, markersize=5, label="production Marlin")
        ax.plot(hs, tt, "o-", color=HBM, linewidth=2, markersize=5, label="tiered kernel")
        for h, m, t in zip(hs, mt, tt):
            if h in (5, 9, 13):   # in the gap between the curves, or above Marlin when it is narrow
                ax.text(h, (m + t) / 2 if m - t > 25 else m + 14, f"{m / t:.2f}x", ha="center", va="center", fontsize=9, color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.9))
        ax.set_title(f"{c} cold expert{'s' if c != 1 else ''}", fontsize=11, color=INK)
        ax.set_xlabel("hot experts on this GPU")
        ax.set_ylim(0, 240)
        ax.set_xlim(-0.5, 16.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        if ci == 0:
            ax.set_ylabel("MoE layer time, w13 + w2 (µs)")
            ax.legend(loc="upper left", fontsize=9, frameon=False)

    # --- heatmap of speedup
    ax = fig.add_subplot(gs[1, :3])
    hot_axis = sorted({h for (h, c) in tiered if c > 0} | {h for (h, c) in tiered if c == 0 and h > 0})
    cold_axis = [0, 1, 2, 3, 4]
    grid = np.full((len(cold_axis), len(hot_axis)), np.nan)
    for yi, c in enumerate(cold_axis):
        for xi, h in enumerate(hot_axis):
            if (h, c) in tiered and (h > 0 or c > 0):
                grid[yi, xi] = marlin(h, c) / tiered[(h, c)]
    cmap = LinearSegmentedColormap.from_list("speed", ["#eef4fc", "#9cc3ee", HBM, "#123e73"])
    ax.imshow(grid, cmap=cmap, vmin=1.0, vmax=1.7, aspect="auto", origin="lower")
    for yi, c in enumerate(cold_axis):
        for xi, h in enumerate(hot_axis):
            v = grid[yi, xi]
            if np.isnan(v):
                continue
            ax.text(xi, yi + 0.08, f"{v:.2f}x", ha="center", va="center", fontsize=10,
                    color="white" if v > 1.35 else INK, fontweight="bold")
            p = prob.get((h, c))
            if p:
                ax.text(xi, yi - 0.28, f"{p:.1%} of layers", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 1.35 else INK2)
            if p and p >= 0.02:
                ax.add_patch(Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))
    ax.set_xticks(range(len(hot_axis)), [str(h) for h in hot_axis])
    ax.set_yticks(range(len(cold_axis)), [str(c) for c in cold_axis])
    ax.set_xlabel("hot experts on this GPU (HBM)")
    ax.set_ylabel("cold experts (Grace, C2C)")
    ax.grid(False)
    ax.set_title("Speedup over Marlin per (hot, cold) cell; outlined: cells with ≥ 2% of real layer calls",
                 fontsize=11, color=INK, loc="left")

    # --- weighted summary
    ax = fig.add_subplot(gs[1, 3])
    wp = sum(r["p"] for r in weighted)
    wm = sum(r["p"] * r["marlin_us"] for r in weighted) / wp
    wt = sum(r["p"] * r["tiered_us"] for r in weighted) / wp
    bars = ax.bar([0, 1], [wm, wt], color=[MARLIN, HBM], width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, [wm, wt]):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f} µs", ha="center", fontsize=11, color=INK)
    ax.set_xticks([0, 1], ["Marlin", "tiered"])
    ax.set_ylim(0, 185)
    ax.set_ylabel("µs per MoE layer, trace-weighted")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_title(f"{wm / wt:.2f}x faster on average", fontsize=11, color=INK)
    ax.text(0.5, -0.2, f"≈ {(wm - wt) * 69 / 1000:.1f} ms saved per decode step\n(69 MoE layers, ~21 ms step)",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, color=INK2)

    fig.suptitle("MiMo-V2.6 MoE layer on one GH200: one-kernel tiered MXFP4 GEMM vs production Marlin",
                 x=0.06, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.985)
    fig.text(0.06, 0.945, "Pinned sustained clock (~1.45 GHz), 8-token decode verify, decode token mix. "
             "Marlin = max(hot, cold) tiers with perfect overlap (its best case); includes act + sum kernels.",
             fontsize=9.5, color=MUTED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
