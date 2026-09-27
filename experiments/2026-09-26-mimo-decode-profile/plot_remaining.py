#!/usr/bin/env python3
"""Where a MiMo-V2.6 decode step still loses time, after the one-kernel MoE.

    plot_remaining.py --out figs/remaining-2092055.png

Numbers are job 2092055 (one-kernel MoE, PDL off for clean kernel durations,
short context, chat prompts, mean of 4 GPUs, 20.9 ms profiled step): the
step breakdown (breakdown-2092055.txt), the cross-rank imbalance
(imbalance-2092055.txt) and the byte floors (sol-2092055.txt). The MoE floor
is E[max(hot bytes / HBM, cold bytes / C2C)] per layer from the expert counts
recorded on the same prompts in job 2077518.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-routing-profile"))
from plot_routing import DDR, GRID, HBM, INK, INK2, MUTED, SURFACE  # noqa: E402

LATENCY = "#c3c2b7"
# (part, measured ms/step, floor ms/step or None, kind, note)
ROWS = [
    ("MoE GEMMs, one kernel (w13 + w2)", 8.79, 6.16, "bw", "70% of the hot/cold floor"),
    ("waiting for the slowest GPU's MoE", 1.64, 0.0, "wait", "all-reduce wait = MoE imbalance"),
    ("GPU idle inside the step", 1.05, 0.0, "wait", "CPU / launch gaps"),
    ("replica assign + align, 1 CTA x69", 0.95, None, "lat", "also builds Marlin metadata the new path ignores"),
    ("o_proj, bf16 (unquantized)", 1.32, 0.96, "bw", "73%; fp8 would halve the floor"),
    ("MoE route / act / finalize + topk", 1.09, None, "lat", "4 small kernels x69"),
    ("TP all-reduce transfer x141", 0.90, 0.61, "lat", "cross-rank minimum 0.61"),
    ("qkv_proj, fp8", 1.06, 0.80, "bw", "76%"),
    ("host-side kernels (sampling, prep)", 0.75, None, "lat", ""),
    ("DFlash drafter", 0.76, None, "lat", "latency-bound small graphs"),
    ("attention (sliding 60 + full 10)", 0.73, None, "lat", "1.25 ms at 96K, full attn 53%"),
    ("norms / rope / elementwise", 0.65, None, "lat", "250 launches"),
    ("router gate, bf16", 0.34, 0.09, "bw", "26%"),
    ("lm_head x3 + logits", 0.68, 0.39, "bw", "lm_head 98%; rest is all-gather"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = sorted(ROWS, key=lambda r: -(r[1] - (r[2] if r[2] is not None else r[1] * 0.5)))
    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor=SURFACE)
    y = np.arange(len(rows))[::-1]
    colors = {"bw": HBM, "wait": DDR, "lat": LATENCY}
    for yi, (name, ms, floor, kind, note) in zip(y, rows):
        ax.barh(yi, ms, height=0.62, color=colors[kind], edgecolor=SURFACE, linewidth=1)
        if floor is not None:
            ax.barh(yi, floor, height=0.62, color=INK, alpha=0.18, edgecolor="none")
            ax.plot([floor, floor], [yi - 0.36, yi + 0.36], color=INK, linewidth=1.8)
            label = f"{ms:.2f} ms  (gap {ms - floor:.2f})"
        else:
            label = f"{ms:.2f} ms"
        if note:
            label += f"  - {note}"
        ax.text(ms + 0.08, yi, label, va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 15.5)
    ax.set_xlabel("ms per decode step (mean of 4 GPUs, short context, 8-token verify)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=HBM, label="bandwidth-bound work"),
                       Line2D([], [], color=INK, linewidth=1.8, label="floor (bytes / roof); shaded"),
                       Patch(color=DDR, label="waiting / idle (floor 0)"),
                       Patch(color=LATENCY, label="small kernels: latency-bound")],
              loc="lower right", fontsize=9, frameon=False)
    ax.set_title("MiMo-V2.6 decode after the one-kernel MoE: 20.9 ms/step, what is left",
                 fontsize=13, fontweight="bold", color=INK, loc="left", pad=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
