#!/usr/bin/env python3
"""Write-up figures for the MLA KV offloads (glm/kv-offload/README.md).

    plot_kvoff.py <out dir>

kv-offload-step.png   one anchor and its skip layers in a decode step: which
                      copies run where (schematic, not to scale)
kv-offload-hbm.png    HBM per GPU per 100K pooled tokens, and hot experts per
                      GPU vs pool size, for A / B / C (planner residency lines,
                      probe_plan.sh and launch_pool.sh)
kv-offload-cost.png   verify step at an equal hot set (launch_eq.sh) and at each
                      arm's own hot set (launch_buy.sh), paired with A per node
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-08-m32"))
from plot_m32 import DDR, GRID, HBM, INK, INK2, MUTED, SURFACE, finish, style  # noqa: E402

COMPUTE = "#eceae4"
CFG = {"A": ("A: all KV in HBM", MUTED), "B": ("B: skip layers on Grace", "#6250d6"),
       "C": ("C: + anchors on Grace (prod)", "#1baf7a")}
EXPERT_GIB = 5308416 / 2**30  # one INT4 expert slice (weights + scales) per GPU
# hot experts per GPU (planner residency line, TP0) by KV pool (M tokens):
# probe_plan.sh via launch_plan.sh (A, B), launch_pool.sh akp* (C)
HOT = {"A": {0.8: 12831, 1.6: 10715}, "B": {0.8: 14284, 1.6: 13609},
       "C": {0.8: 14773, 1.2: 14702, 1.6: 14631, 2.4: 14488, 3.2: 14345}}


def box(ax, x, y, w, h, text, face, edge=None, ink=INK, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=face, edgecolor=edge or face, linewidth=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=ink)


def arrow(ax, a, b, color=INK2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=color,
                                 linewidth=1.2, shrinkA=0, shrinkB=0))


def step(out):
    fig, ax = plt.subplots(figsize=(12, 3.9), facecolor=SURFACE)
    ax.set_xlim(-1.9, 13.2)
    ax.set_ylim(-0.62, 2.48)
    ax.axis("off")
    main, side = 1.75, 0.45
    ax.text(-1.85, main + 0.3, "main\nstream", fontsize=10, color=INK2, va="center")
    ax.text(-1.85, side + 0.3, "side\nstream", fontsize=10, color=INK2, va="center")
    for y in (main, side):
        ax.plot([-0.9, 13.1], [y - 0.08] * 2, color=GRID, linewidth=0.8)

    h = 0.6
    box(ax, -0.8, main, 1.5, h, "indexer\n-> top-k", COMPUTE)
    box(ax, 0.8, main, 1.4, h, "KV write,\nq absorb", COMPUTE)
    box(ax, 2.3, main, 1.7, h, "FlashMLA\n(hot buffer)", SURFACE, HBM)
    box(ax, 4.1, main, 1.3, h, "o_proj,\nMoE", COMPUTE)
    for i, x in enumerate((5.5, 8.1, 10.7)):
        box(ax, x, main, 1.5, h, f"FlashMLA\n(staged rows)", SURFACE, HBM)
        if i < 2:
            box(ax, x + 1.6, main, 0.9, h, "MoE", COMPUTE)
    box(ax, 0.75, side, 1.5, h, "map: misses\nGrace -> HBM", DDR, ink="white")
    box(ax, 2.35, side, 2.6, h, "stage the skip layers' rows\nGrace -> HBM", DDR,
        ink="white")
    arrow(ax, (0.7, main + 0.05), (0.85, side + h))
    arrow(ax, (2.25, side + h), (2.45, main))
    arrow(ax, (3.6, main), (3.7, side + h))
    arrow(ax, (4.95, side + h / 2), (5.55, main))

    ax.text(-0.05, main + h + 0.1, "anchor layer (own indexer)", fontsize=10, color=INK,
            fontweight="bold", ha="left")
    ax.text(5.5, main + h + 0.1, "3 skip layers (reuse the anchor's top-k)", fontsize=10,
            color=INK, fontweight="bold", ha="left")
    ax.text(0.75, side - 0.12,
            "~93% of the rows are already in the anchor's\n16K-row HBM buffer; "
            "~230 per call are copied\n(~11 us, hidden under the KV write)",
            fontsize=8.5, color=INK2, va="top")
    ax.text(5.15, side - 0.12,
            "the rows the anchor's top-k picked, for all three;\ndone before the first skip "
            "layer needs them",
            fontsize=8.5, color=INK2, va="top")
    finish(fig, "Every layer still reads HBM: the copies run beside the main stream",
           "One of the 21 anchor groups in a decode step (schematic, not to scale). "
           "Blue outline: reads HBM. Orange: reads Grace over C2C.",
           out / "kv-offload-step.png", top=0.84)


def slope(cfg):
    """HBM per GPU per 100K pooled tokens (GiB), from the planner's hot counts."""
    pts = sorted(HOT[cfg].items())
    (p0, h0), (p1, h1) = pts[0], pts[-1]
    return (h0 - h1) * EXPERT_GIB / ((p1 - p0) * 10)


def hbm(out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    k = {c: slope(c) for c in CFG}
    for i, c in enumerate(CFG):
        ax.bar(i, k[c], width=0.6, color=CFG[c][1], edgecolor=SURFACE)
        ax.text(i, k[c] + 0.03, f"{k[c]:.2f} GiB", ha="center", fontsize=10, color=INK,
                fontweight="bold")
    ax.annotate("", (1, k["B"] + 0.15), (0.3, k["A"] - 0.1),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    ax.text(0.72, (k["A"] + k["B"]) / 2 + 0.12, f"-{k['A'] - k['B']:.2f}\nskip-layer KV",
            fontsize=9, color=INK2)
    ax.annotate("", (2, k["C"] + 0.15), (1.3, k["B"] + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    ax.text(1.72, (k["B"] + k["C"]) / 2 + 0.12, f"-{k['B'] - k['C']:.2f}\nanchor KV",
            fontsize=9, color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["A", "B", "C"], color=INK)
    ax.set_ylim(0, 1.55)
    ax.set_ylabel("HBM per GPU per 100K pooled tokens (GiB)", color=INK2)
    ax.set_title("What the KV pool costs in HBM", fontsize=11, color=INK)
    style(ax)

    for c, (label, color) in CFG.items():
        pts = sorted(HOT[c].items())
        x, y = zip(*pts)
        bx.plot(x, y, color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, zorder=3)
        bx.text(x[-1] + 0.12, y[-1] + (90 if c == "B" else 0), label, color=INK, fontsize=9.5,
                va="center")
    a0 = HOT["A"][0.8]
    bx.axhline(a0, color=GRID, lw=1.2, ls="--", zorder=1)
    reach_b = 0.8 + (HOT["B"][0.8] - a0) * EXPERT_GIB / k["B"] / 10
    bx.plot([1.6, reach_b], [HOT["B"][1.6], a0], color=CFG["B"][1], lw=1.5, ls=":")
    bx.plot([reach_b], [a0], marker="o", ms=7, mfc=SURFACE, mec=CFG["B"][1], mew=2)
    bx.text(reach_b, a0 - 230, f"B reaches A's hot set\nat ~{reach_b:.1f}M (extrapolated)",
            fontsize=8.5, color=INK2, ha="center", va="top")
    bx.text(4.15, a0 + 60, f"A's hot set at 0.8M: {a0:,}", fontsize=8.5, color=INK2,
            ha="right")
    bx.set_xlim(0.4, 4.2)
    bx.set_ylim(10300, 15200)
    bx.set_xlabel("KV pool (M tokens, all GPUs)", color=INK2)
    bx.set_ylabel("hot experts per GPU (of 19,200)", color=INK2)
    bx.set_title("Hot experts as the pool grows", fontsize=11, color=INK)
    style(bx, "both")
    finish(fig, "The pool now costs HBM 14x less: 1.31 -> 0.09 GiB per 100K tokens",
           "Planner residency at the c=4 config. What stays in HBM in C: the indexer's keys, "
           "the MTP layer's KV, the slot tables.\nC's pool is bounded by Grace, not HBM: at "
           "3.2M (the largest tested) GPU 0's Grace holds 77 of ~119 GiB.",
           out / "kv-offload-hbm.png", top=0.8)


def cost(out):
    eq = json.loads((HERE / "ak_pairs.json").read_text())
    buy = json.loads((HERE / "akb_pairs.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE, sharey=True)
    panels = [(axes[0], eq, {"y": "B", "z": "C"}, "rest",
               "Same hot experts (12,831): the cost"),
              (axes[1], buy, {"u": "B", "w": "C"}, "span",
               "Same 0.8M pool: with the freed HBM as hot experts")]
    for ax, pairs, arms, field, title in panels:
        xs, labels = [], []
        for j, (arm, c) in enumerate(arms.items()):
            for i, win in enumerate((1, 2)):
                x = j * 2.6 + i
                v = [p[field] / 1e3 for p in pairs if p["arm"] == arm and p["win"] == win]
                m = sum(v) / len(v)
                ax.bar(x, m, width=0.75, color=CFG[c][1], edgecolor=SURFACE)
                ax.scatter([x] * len(v), v, s=16, color=INK, alpha=0.55, zorder=3, lw=0)
                ax.text(x, max(v) + 0.1 if m > 0 else min(v) - 0.1, f"{m:+.2f}", ha="center",
                        va="bottom" if m > 0 else "top", fontsize=10, color=INK,
                        fontweight="bold")
                xs.append(x)
                labels.append(f"{c} - A\n{'~5K' if win == 1 else '~50K'}")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, color=INK, fontsize=9.5)
        ax.set_title(title, fontsize=11, color=INK)
        style(ax)
    axes[0].set_ylabel("verify step vs A, same node (ms)", color=INK2)
    axes[0].set_ylim(-4.4, 1.2)
    axes[0].text(0.02, 0.04, "step time outside the MoE\n(the MoE varies with the\ntraffic's "
                 "cold experts)", transform=axes[0].transAxes, fontsize=8.5, color=INK2)
    axes[1].text(0.02, 0.04, "whole verify step", transform=axes[1].transAxes, fontsize=8.5,
                 color=INK2)
    finish(fig, "Offloading costs ~0.4-0.5 ms per step; the hot experts it buys save 2-3 ms",
           "c=4, MTP3 (16-token verify steps), 4 requests at ~5K / ~50K context; "
           "verify step ~24.5 ms in A. Bars: mean of same-node pairs (dots: one per node, "
           "6 / 3 nodes).",
           out / "kv-offload-cost.png", top=0.82)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    step(out)
    hbm(out)
    cost(out)


if __name__ == "__main__":
    main()
