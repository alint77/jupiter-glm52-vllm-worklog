#!/usr/bin/env python3
"""Figures explaining the hot/cold Marlin overlap, for the tiered-MoE write-up.

    plot_overlap.py --out gist/

concept   idealized MoE layer time vs share of expert bytes read from Grace,
          serial vs overlapped tiers, from measured achieved bandwidths
smem      one SM's shared memory under the stock and the tight Marlin launch
sweep     measured two-tier union before/after the fix (Booster, graph replay)
ladder    GLM-5.2 exact-400K batch-one decode across the project's phases
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_routing import DDR, GRID, HBM, INK, INK2, MUTED, SURFACE

EXP = Path("/e/project1/profound/alint77/vllm/agent_space/experiments")
B_HBM = 2150.0  # hot Marlin achieved GB/s (2026-07-25-grace-bandwidth)
B_C2C = 410.0  # cold Marlin / C2C achieved GB/s, NUMA-local


def fig_concept(out: Path) -> None:
    x = np.linspace(0, 0.5, 501)
    hot = (1 - x) / B_HBM
    cold = x / B_C2C
    base = 1 / B_HBM
    serial = (hot + cold) / base
    overlap = np.maximum(hot, cold) / base
    balance = B_C2C / (B_C2C + B_HBM)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(1, color=MUTED, linewidth=1.5, linestyle=":")
    ax.text(49.5, 1.03, "everything in HBM", ha="right", va="bottom", color=INK2, fontsize=10)
    ax.plot(x * 100, serial, color=DDR, linewidth=2, label="tiers run one after the other")
    ax.plot(x * 100, overlap, color=HBM, linewidth=2, label="tiers run at the same time")
    ax.axvline(balance * 100, color=INK2, linewidth=0.8, linestyle="--")
    ax.text(balance * 100 + 0.6, 2.5, f"balance point ~{balance * 100:.0f}%:\nboth tiers finish together",
            color=INK2, fontsize=10, va="top")
    ax.fill_between(x * 100, overlap, 1, where=overlap <= 1, color=HBM, alpha=0.15, linewidth=0)
    ax.annotate("below 1.0, offloading is free:\nGrace time hides under HBM time", xy=(9, 0.95),
                xytext=(20.5, 0.66), color=HBM, fontsize=10,
                arrowprops={"arrowstyle": "-", "color": HBM, "linewidth": 1})
    ax.set_xlim(0, 50)
    ax.set_ylim(0.6, 3.2)
    ax.set_xlabel("share of a step's expert bytes read from Grace (%)")
    ax.set_ylabel("MoE time vs. all-in-HBM (x)")
    ax.set_title("Overlap is what makes offloading cheap\n"
                 "idealized, from measured HBM 2.15 / C2C 0.41 TB/s",
                 pad=10, linespacing=1.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "overlap-concept.png", dpi=160)
    plt.close(fig)


def fig_smem(out: Path) -> None:
    kib = 1024
    sm = 233_472 / kib
    stock = 76_458 / kib
    tight = 25_856 / kib
    reserved = 1.0
    fig, ax = plt.subplots(figsize=(9, 3.2))

    def cta(y, left, width, color, label, text_color="white"):
        ax.barh(y, width, left=left, color=color, height=0.5, edgecolor=SURFACE, linewidth=2)
        ax.text(left + width / 2, y, label, ha="center", va="center", color=text_color, fontsize=9)
        return left + width + reserved

    left = 0.0
    for _ in range(3):
        left = cta(1, left, stock, HBM, "hot CTA\n74.7 KiB")
    ax.barh(1, stock, left=sm + 4, color="none", height=0.5, edgecolor=DDR, linewidth=1.5,
            hatch="///", linestyle="--")
    ax.text(sm + 4 + stock / 2, 1, "cold CTA\ndoesn't fit", ha="center", va="center", color=INK,
            fontsize=9)
    left = 0.0
    for _ in range(2):
        left = cta(0, left, tight, HBM, "hot\n25 KiB")
    left = cta(0, left, tight, DDR, "cold\n25 KiB", INK)
    ax.text(left + (sm - left) / 2, 0, f"{sm - left:.0f} KiB free", ha="center", va="center",
            color=INK2, fontsize=10)
    ax.axvline(sm, color=INK, linewidth=1.2)
    ax.text(sm, 1.45, "one SM: 228 KiB", ha="right", va="bottom", color=INK, fontsize=10)
    ax.set_yticks([1, 0], ["stock launch:\nasks 3x what it uses", "tight launch:\nasks what it uses"])
    ax.set_xlim(0, sm + stock + 8)
    ax.set_ylim(-0.5, 1.75)
    ax.set_xlabel("shared memory on one SM (KiB)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Why the tiers never overlapped", pad=10)
    fig.tight_layout()
    fig.savefig(out / "overlap-smem.png", dpi=160)
    plt.close(fig)


def fig_sweep(out: Path) -> None:
    data = json.loads((EXP / "2026-07-29-marlin-smem-monopoly/booster-kernel-ab-graph.json").read_text())
    rows = []
    for row in data["rows"]:
        layers = row["layers"]

        def total(key):
            return sum(layers[name][key] for name in ("w13", "w2"))

        hot, cold = total("g_hot_solo"), total("g_cold_solo")
        rows.append((row["m"], row["hot"], row["cold"], hot + cold, total("g_prod"),
                     total("g_tight"), max(hot, cold)))
    rows.sort(key=lambda r: (r[2] / (r[1] + r[2]), r[1] + r[2]))
    fig, ax = plt.subplots(figsize=(9, 6))
    for y, (m, h, c, serial, prod, tight, ideal) in enumerate(rows):
        ax.plot([tight, prod], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.plot(prod, y, "o", color=MUTED, markersize=8, zorder=2)
        ax.plot(tight, y, "o", color=HBM, markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3)
        ax.plot(ideal, y, "|", color=INK, markersize=14, markeredgewidth=2, zorder=4)
        ax.text(prod + 8, y, f"-{(1 - tight / prod) * 100:.0f}%", va="center", color=INK2,
                fontsize=9)
    ax.set_yticks(range(len(rows)),
                  [f"{h} hot + {c} cold, {m} tok" for m, h, c, *_ in rows], fontsize=9)
    ax.plot([], [], "o", color=MUTED, label="stock launch (tiers barely overlap)")
    ax.plot([], [], "o", color=HBM, label="tight launch")
    ax.plot([], [], "|", color=INK, markersize=12, markeredgewidth=2,
            label="max(hot alone, cold alone)")
    ax.set_xlabel("one MoE layer, both tiers (us, CUDA-graph replay)")
    ax.set_title("Measured: with the fix, a layer costs about max(hot, cold)\n"
                 "GH200 (Booster), GLM expert shapes, sorted by cold share",
                 pad=10, linespacing=1.5)
    ax.legend(loc="upper left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "overlap-sweep.png", dpi=160)
    plt.close(fig)


def fig_ladder(out: Path) -> None:
    steps = [
        ("stock vLLM\nCPU offload", 37.57, False),
        ("tiered,\nfirst version", 36.57, True),
        ("MoE comm\npath fix", 55.16, False),
        ("+ MTP3\n(tiers serial)", 108.17, False),
        ("+ hot/cold\noverlap", 127.67, True),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(steps))
    colors = [HBM if offload else "#c3c2b7" for *_, offload in steps]
    ax.bar(x, [v for _, v, _ in steps], color=colors, width=0.62, edgecolor=SURFACE, linewidth=2)
    for xi, (_, value, _) in zip(x, steps):
        ax.text(xi, value + 2, f"{value:.0f}", ha="center", va="bottom", color=INK, fontsize=11)
    ax.set_xticks(x, [name for name, *_ in steps], fontsize=10)
    ax.set_ylabel("decode tok/s")
    ax.set_ylim(0, 145)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=HBM, label="offload work"),
                       Patch(color="#c3c2b7", label="everything else")], loc="upper left")
    ax.set_title("GLM-5.2 (361 GiB) on one 4x GH200 node, 400K context, batch one", pad=10)
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_concept(args.out)
    fig_smem(args.out)
    fig_sweep(args.out)
    fig_ladder(args.out)


if __name__ == "__main__":
    main()
