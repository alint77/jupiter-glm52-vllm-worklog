#!/usr/bin/env python3
"""A simplified Gantt of one DFlash2 decode step (gantt_data.py output), rank 0.

Three panels: the whole step with same-category kernels merged into runs; one
MoE layer's worth of verify-graph kernels, unmerged, so the gaps between them
show; and the tail (sampling + eager drafter) with its idle gaps, marked when the GPU
was waiting on a CPU launch.   plot_gantt.py <gantt json> <out.png>
"""

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-routing-profile"))
from plot_routing import INK, INK2, SURFACE  # noqa: E402

# dataviz reference categorical palette, light, in slot order
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
IDLE = "#d6d4ce"


def merge(kernels, tol_us):
    """Consecutive kernels of one category, gaps under tol_us, as one run."""
    runs = []
    for s, d, b, *_ in kernels:
        if runs and runs[-1][2] == b and s - (runs[-1][0] + runs[-1][1]) < tol_us:
            runs[-1][1] = max(runs[-1][1], s + d - runs[-1][0])
        else:
            runs.append([s, d, b])
    return runs


def lane(ax, y, items, h=0.7, color_of=lambda it: CAT[it[2]], scale=1.0):
    for it in items:
        ax.broken_barh([(it[0] * scale, max(it[1] * scale, 0))], (y - h / 2, h),
                       facecolors=color_of(it), linewidth=0)


def main() -> None:
    d = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    r0 = d["ranks"]["0"]
    step_end = d["steps"][1]["start"]
    kern = [k for k in r0["kernels"] if k[0] < step_end]
    main_id = max({k[4] for k in kern}, key=lambda s: sum(1 for k in kern if k[4] == s))
    main_k = [k for k in kern if k[4] == main_id]
    side_k = [k for k in kern if k[4] != main_id]
    gaps = [g for g in r0["gaps"] if g[0] < step_end]
    host = [h for h in r0["host"] if h[0] < step_end]
    ph = {}
    for k in kern:
        e = ph.setdefault(k[5], [k[0], k[0] + k[1]])
        e[0], e[1] = min(e[0], k[0]), max(e[1], k[0] + k[1])
    ver, samp, draft = ph["target"], ph["logits"], ph["draft"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2),
                             gridspec_kw={"height_ratios": [1.05, 1, 1.15], "hspace": 0.75})
    fig.patch.set_facecolor(SURFACE)

    # (a) whole step, ms
    ax = axes[0]
    ms = 1 / 1000
    lane(ax, 1, merge(main_k, 3.0), scale=ms)
    lane(ax, 0, merge(side_k, 3.0), h=0.45, scale=ms)
    a, b = ver
    ax.annotate("", xy=(a * ms, 1.62), xytext=(b * ms, 1.62),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text((a + b) / 2 * ms, 1.78,
            f"verify: one CUDA graph, ~3,400 kernels\n{(b - a) / 1000:.2f} ms",
            ha="center", va="bottom", fontsize=9, color=INK)
    ax.annotate("", xy=(samp[0] * ms, 1.62), xytext=(draft[1] * ms, 1.62),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text(draft[1] * ms, 1.78,
            f"sample {(samp[1] - samp[0]) / 1000:.2f} ms\n"
            f"+ drafter {(draft[1] - draft[0]) / 1000:.2f} ms",
            ha="right", va="bottom", fontsize=9, color=INK)
    ax.set_yticks([1, 0], ["GPU main stream", "side stream"])
    ax.set_xlim(0, step_end * ms)
    ax.set_ylim(-0.5, 2.55)
    ax.set_xlabel("ms into the step")
    ax.set_title(f"One decode step, rank 0: {d['steps'][0]['period_ms']:.1f} ms",
                 loc="left", fontsize=12)

    # (b) one layer inside the verify graph, us
    ax = axes[1]
    mid = ver[0] + 0.45 * (ver[1] - ver[0])
    a, b = mid, mid + 330
    sel = lambda ks: [k for k in ks if k[0] + k[1] > a and k[0] < b]  # noqa: E731
    lane(ax, 1, [[k[0] - a, k[1], k[2]] for k in sel(main_k)])
    lane(ax, 0, [[k[0] - a, k[1], k[2]] for k in sel(side_k)], h=0.45)
    g = [x for x in gaps if a < x[0] < b]
    idle = sum(x[1] for x in g)
    ax.set_yticks([1, 0], ["GPU main stream", "side stream"])
    ax.set_xlim(0, b - a)
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("µs")
    ax.set_title(f"Zoom: ~one MoE layer of the verify graph. {len(g)} idle gaps, "
                 f"{idle:.0f} µs in total, almost all under 1 µs between kernels",
                 loc="left", fontsize=12)

    # (c) tail: sampling + drafter, with CPU launches
    ax = axes[2]
    a, b = samp[0] - 60, draft[1] + 40
    lane(ax, 1, [[k[0] - a, k[1], k[2]] for k in sel(main_k) if a <= k[0] <= b])
    g = [x for x in gaps if a < x[0] < b]
    lane(ax, 0, [[x[0] - a, x[1], x[2]] for x in g], h=0.45,
         color_of=lambda it: CAT[7] if it[2] else IDLE)
    starved = sum(x[1] for x in g if x[2])
    ax.set_yticks([1, 0], ["GPU main stream", "idle gaps"])
    ax.set_xlim(0, b - a)
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("µs from the end of the verify graph")
    ax.set_title(f"Zoom: sampling and the eager drafter. {len(g)} idle gaps, "
                 f"{sum(x[1] for x in g):.0f} µs; waiting on the CPU: {starved:.0f} µs",
                 loc="left", fontsize=12)

    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)

    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(CAT, d["buckets"])]
    handles.append(mpatches.Patch(color=IDLE, label="idle gap (red if waiting on the CPU)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("GLM-5.3 with DFlash2 (DCP4, eager drafter): where one decode step goes",
                 x=0.01, ha="left", fontweight="bold", fontsize=13, color=INK)
    fig.subplots_adjust(top=0.9, bottom=0.14, left=0.12, right=0.98)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
