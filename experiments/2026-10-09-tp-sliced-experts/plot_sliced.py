#!/usr/bin/env python3
"""Write-up figure: throughput per GPU vs interactivity, prod EP vs TP-sliced
MoE, MTP3 at c=8 on the 1.6M pool (sweep_ab.py's arms, chain_sweep.sh).

    plot_sliced.py <out dir>      -> glm-sliced-concurrency.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-08-m32"))
from plot_m32 import DDR, GPUS, HBM, INK, INK2, MUTED, SURFACE, style  # noqa: E402
from sweep_ab import collect  # noqa: E402

KINDS = [("ep", "EP MoE (prod: whole experts per GPU, 2,000 Grace replicas)", DDR, "--", "s"),
         ("sl", "TP-sliced MoE (every GPU a quarter of every expert)", HBM, "-", "o")]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    arms, agg, _ = collect()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), facecolor=SURFACE, sharey=True)
    for ax, ctx in zip(axes, (5000, 50000)):
        ns = sorted(n for c, n in agg["ep"] if c == ctx)
        pts = {k: [(agg[k][(ctx, n)]["user"], agg[k][(ctx, n)]["gpu"]) for n in ns] for k, *_ in KINDS}
        for (xe, ye), (xs, ys) in zip(pts["ep"], pts["sl"]):
            ax.annotate("", (xs, ys), (xe, ye), arrowprops=dict(arrowstyle="-|>", color=MUTED,
                        lw=1.0, shrinkA=6, shrinkB=6, alpha=0.8), zorder=2)
        for k, label, color, ls, mk in KINDS:
            ax.plot([x for x, _ in pts[k]], [y for _, y in pts[k]], ls, marker=mk, color=color,
                    markersize=7, linewidth=2.2, label=label, zorder=3)
        for n, (xe, ye), (xs, ys) in zip(ns, pts["ep"], pts["sl"]):
            ax.annotate(f"{n}", (xs, ys), xytext=(7, 4), textcoords="offset points", fontsize=10,
                        color=INK, fontweight="bold")
            ax.annotate(f"{(ys / ye - 1) * 100:+.0f}%", (xs, ys), xytext=(8, -16),
                        textcoords="offset points", fontsize=8.5, color=HBM,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"), zorder=4)
        ax.set_title(f"{ctx // 1000}K tokens of context per request", loc="left", fontsize=10.5,
                     color=INK)
        ax.set_xlabel("interactivity: decode tok/s per user", color=INK2)
        ax.set_xlim(50, 210)
        style(ax, "both")
    axes[0].set_ylabel(f"throughput: output tok/s per GPU ({GPUS} GH200)", color=INK2)
    axes[0].set_ylim(0, 145)
    axes[0].text(0.98, 0.97, "bold: requests in flight\nblue %: throughput gain",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    n_arms = min(len(v) for v in arms.values())
    gain = [agg["sl"][k]["gpu"] / agg["ep"][k]["gpu"] - 1 for k in agg["ep"]]
    fig.suptitle(f"TP-sliced MoE lifts the whole MTP3 curve: {min(gain) * 100:.0f}-"
                 f"{max(gain) * 100:.0f}% more throughput and interactivity at every load", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.905, f"GLM-5.3 W4A16, TP4 / DCP4, MTP3, c=8 on a shared 1.6M KV pool; "
             f"400 output tokens, temperature 1.0; {n_arms} servers per layout, same nodes",
             fontsize=9.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out / "glm-sliced-concurrency.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
