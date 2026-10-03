#!/usr/bin/env python3
"""Figures for the GLM-5.3 prefill page (gh200-tiered-moe glm/prefill/README.md).

    plot_prefill.py --out <gh200-tiered-moe>/glm/prefill/figs

ttft       TTFT vs prompt length: no prefetch, Marlin, the prefill kernel
moe        whole routed MoE per layer at 2K / 4K tokens across the kernel work
breakdown  one 4K-token chunk at the agentic shape, by kernel category

Numbers are the measured ones in README.md (this folder); see the comments.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26-mimo-routing-profile"))
from plot_routing import DDR, GRID, HBM, INK, INK2, MUTED, SURFACE  # noqa: E402,F401

# Median TTFT ms, 20 random prompts, same node (launch_pk.sh node A; the
# no-prefetch arm from the in-graph prefetch A/B, jpbo-001-44).
TTFT_LEN = [512, 768, 1024, 2048, 4096]
TTFT = {
    "no cold prefetch, Marlin": [176, 236, 240, None, None],
    "in-graph prefetch, Marlin": [134, 171, 202, 327, 990],
    "in-graph prefetch, prefill kernel": [122, 135, 154, 243, 873],
}

# Whole routed MoE on one GPU, real GLM-5.3 routing (bench_iter.py, 4 chunks),
# us per layer at 2048 / 4096 tokens; Marlin from bench_grouped_real (16 chunks).
MARLIN = (2064, 3365)
MOE = [
    ("first integrated\nkernel", 1363, 2344),
    ("96-row tiles,\neven splits", 1300, 2156),
    ("scales off the\nstack, bigger rounds", 1207, 1984),
    ("producer\nwarpgroups", 1192, 1956),
    ("epilogue scales\nloaded up front", 1123, 1841),
    ("epilogue via smem,\n16 B glue kernels", 1095, 1792),
]

# One 4089-token chunk (20K new on 14K cached), rank 0, prof_window.py union
# ms per category over 4 traced chunks. Runs 4-5 are one node (fused combine
# off / on); the first three are earlier runs.
CATS = ["MoE", "collectives", "DCP correction", "DCP layout copies",
        "dense GEMMs", "sparse attention", "indexer", "norms, elementwise, misc",
        "idle"]
COLORS = [HBM, DDR, "#f2a07b", "#c9c7c1", "#5a9be3", "#7f5fc2", "#3aa17e", MUTED,
          GRID]
RUNS = [  # label, per-category ms
    ("start: prefill kernel,\nsparse prefill, 4K chunks",
     [171.8 + 16.6 + 1.8, 192.6, 16.6, 86.1, 75.4, 59.9, 12.2 + 1.5,
      14.3 + 11.7 + 6.2 + 1.1, 14.4]),
    ("DCP layout\ncopies removed",
     [164.7 + 16.9 + 1.8, 204.0, 17.3, 7.6, 73.3, 61.6, 12.4 + 1.5,
      14.9 + 11.7 + 6.4 + 1.2, 14.1]),
    ("MoE kernel\nwork",
     [125.0 + 13.0 + 1.8, 199.3, 17.6, 7.6, 73.6, 61.0, 12.4 + 1.5,
      14.7 + 11.7 + 6.3 + 1.2, 11.9]),
    ("node B:\nNCCL combine",
     [138.5 + 13.2 + 1.9, 185.1, 17.6, 7.6, 79.3, 63.0, 12.6 + 1.6,
      15.4 + 11.7 + 6.4 + 1.2, 12.8]),
    ("node B:\nfused combine",
     [136.4 + 13.1 + 1.8, 180.0, 0.0, 7.6, 76.9, 62.4, 12.5 + 1.5,
      15.2 + 11.7 + 6.4 + 1.2, 14.9]),
]


def fig_ttft(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(TTFT_LEN))
    w = 0.27
    for i, ((name, vals), color) in enumerate(zip(TTFT.items(), (MUTED, DDR, HBM))):
        xs = [xi + (i - 1) * w for xi, v in zip(x, vals) if v is not None]
        vs = [v for v in vals if v is not None]
        ax.bar(xs, vs, w * 0.95, color=color, label=name)
        for xi, v in zip(xs, vs):
            ax.text(xi, v + 12, f"{v}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [f"{n:,}" for n in TTFT_LEN])
    ax.set_xlabel("prompt tokens (no cache)")
    ax.set_ylabel("median TTFT, ms")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Prefill TTFT: cold prefetch inside the prefill graphs, then the "
                 "INT4 prefill kernel", loc="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out / "glm-prefill-ttft.png", dpi=160)
    plt.close(fig)


def fig_moe(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(MOE))
    for col, (tok, color, marlin) in enumerate(((2048, "#5a9be3", MARLIN[0]),
                                                (4096, HBM, MARLIN[1]))):
        vals = [m[1 + col] for m in MOE]
        ax.plot(x, vals, "o-", color=color, linewidth=2.2, markersize=7,
                markeredgecolor=SURFACE, label=f"{tok:,} tokens")
        for xi, v in zip(x, vals):
            ax.text(xi, v + 70, f"{v:,}", ha="center", fontsize=9, color=INK)
        ax.axhline(marlin, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(len(MOE) - 0.6, marlin + 40, f"Marlin, {tok:,} tokens: {marlin:,}",
                ha="right", fontsize=9, color=INK2)
    ax.set_xticks(x, [m[0] for m in MOE], fontsize=8.5)
    ax.set_ylim(0, 3700)
    ax.set_ylabel("whole routed MoE, us per layer (one GPU)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("The prefill MoE kernel at 2-4K tokens: -20% / -24% on top of the "
                 "first version (real routing)", loc="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out / "glm-prefill-moe.png", dpi=160)
    plt.close(fig)


def fig_breakdown(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y = np.arange(len(RUNS))[::-1]
    left = np.zeros(len(RUNS))
    vals = np.array([r[1] for r in RUNS])
    for ci, (cat, color) in enumerate(zip(CATS, COLORS)):
        ax.barh(y, vals[:, ci], left=left, color=color, edgecolor=SURFACE,
                linewidth=1, height=0.62, label=cat)
        for yi, l, v in zip(y, left, vals[:, ci]):
            if v >= 40:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                        fontsize=8.5, color="white" if ci in (0, 1, 4, 5) else INK)
        left += vals[:, ci]
    for yi, total in zip(y, left):
        ax.text(total + 6, yi, f"{total:.0f} ms", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [r[0] for r in RUNS], fontsize=9)
    ax.set_xlim(0, 760)
    ax.set_xlabel("ms per 4K-token chunk (rank 0)")
    ax.legend(frameon=False, ncol=5, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.45, -0.16))
    ax.set_title("One 4K-token prefill chunk at the agentic shape (20K new on 14K "
                 "cached)", loc="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out / "glm-prefill-breakdown.png", dpi=160)
    plt.close(fig)


FIGS = {"ttft": fig_ttft, "moe": fig_moe, "breakdown": fig_breakdown}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*", choices=list(FIGS))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.only or FIGS:
        FIGS[name](a.out)
