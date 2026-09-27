#!/usr/bin/env python3
"""Figures for the gh200-tiered-moe write-up that follow the one-kernel work.

    plot_writeup.py --out <gh200-tiered-moe>/figs

mimo-ladder.png  MiMo decode step across every change, measured. Each bar is
                 the median step of that change's own "after" runs; the label
                 is the change measured against its own same-node control
                 (servers and nodes differ between A/Bs, so adjacent bars are
                 not exact differences).
step-now.png     where one decode step goes now: profile job 2094684
                 (../2026-09-26-mimo-decode-profile/breakdown-2094684.txt,
                 imbalance-2094684.txt), mean of the 4 GPUs.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-routing-profile"))
from plot_routing import DDR, GRID, HBM, INK, INK2, SURFACE  # noqa: E402

NEUTRAL = "#c3c2b7"

# (label, median step ms, change vs its own control, offload work?)
LADDER = [
    ("arbitrary\nhot set", 34.73, None, True),
    ("hot set from\nrouting traces", 25.03, "-28%", True),
    ("every GPU\nroutes all\ntokens", 21.82, "-13%", False),
    ("+ replicas of\nbusy experts", 20.73, "-5.3%", True),
    ("one kernel\nfor both tiers", 18.37, "-9%", True),
    ("balance GPUs\nby time, in\nthe kernel", 17.13, "-5.6%", True),
]

# (part, ms per step, kind): kind picks the colour
STEP_NOW = [
    ("MoE expert GEMMs (hot + cold, one kernel)", 7.65, "moe"),
    ("dense GEMMs (qkv, o_proj, router)", 2.98, "other"),
    ("waiting for the slowest GPU", 1.56, "wait"),
    ("attention", 1.01, "other"),
    ("GPU idle between launches", 0.98, "wait"),
    ("MoE small kernels (routing, prep, act, sum)", 0.91, "moe"),
    ("DFlash drafter", 0.88, "other"),
    ("host-side ops (input prep, sampling)", 0.75, "other"),
    ("all-reduce data transfer", 0.61, "other"),
    ("norms, rope, elementwise", 0.57, "other"),
    ("logits (lm_head + gather)", 0.28, "other"),
]


def fig_ladder(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.6), facecolor=SURFACE)
    x = np.arange(len(LADDER))
    ax.bar(x, [v for _, v, _, _ in LADDER], width=0.62, edgecolor=SURFACE, linewidth=2,
           color=[HBM if offload else NEUTRAL for *_, offload in LADDER])
    for xi, (_, value, change, offload) in zip(x, LADDER):
        ax.text(xi, value + 0.5, f"{value:.1f} ms", ha="center", va="bottom", color=INK)
        if change:
            ax.text(xi, value / 2, change, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if offload else INK)
    ax.set_xticks(x, [name for name, *_ in LADDER], fontsize=9)
    ax.set_ylabel("decode step, ms (8 tokens verified)")
    ax.set_ylim(0, 42)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=HBM, label="offload work"),
                       Patch(color=NEUTRAL, label="everything else")], loc="upper right",
              frameon=False)
    ax.set_title("MiMo-V2.6 decode on one 4x GH200 node: ~110 -> ~198 tok/s\n"
                 "(inside the bars: each change against its own same-node control)",
                 pad=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "mimo-ladder.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_step_now(out: Path) -> None:
    colors = {"moe": HBM, "wait": DDR, "other": NEUTRAL}
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    y = np.arange(len(STEP_NOW))[::-1]
    for yi, (name, ms, kind) in zip(y, STEP_NOW):
        ax.barh(yi, ms, height=0.62, color=colors[kind], edgecolor=SURFACE, linewidth=1)
        ax.text(ms + 0.08, yi, f"{ms:.2f} ms", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [name for name, _, _ in STEP_NOW], fontsize=9.5)
    ax.set_xlim(0, 9)
    ax.set_xlabel("ms per decode step (mean of 4 GPUs; 18.2 ms profiled)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(handles=[Patch(color=HBM, label="MoE"),
                       Patch(color=DDR, label="waiting / idle"),
                       Patch(color=NEUTRAL, label="everything else")],
              loc="lower right", frameon=False)
    ax.set_title("Where a MiMo decode step goes now", loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out / "step-now.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_ladder(args.out)
    fig_step_now(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
