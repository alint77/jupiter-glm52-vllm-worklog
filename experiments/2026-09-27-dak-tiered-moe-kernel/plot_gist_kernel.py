#!/usr/bin/env python3
"""Gist figure for the one-kernel tiered MoE decode path on MiMo-V2.6.

    plot_gist_kernel.py --out figs/one-kernel.png

Left: one MoE layer (w13 + w2, both tiers) per (hot, cold) expert mix, graph
replay wall clock on one GH200: production two-stream Marlin vs the one kernel
(marlin_wall.py / bench_vllm_decode.py --wall). Right: median decode step of
every end-to-end A/B arm (jobs 2085950 and 2091191).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "2026-09-26-mimo-routing-profile"))
from plot_routing import GRID, HBM, INK, INK2, MUTED, SURFACE  # noqa: E402

sys.path.insert(0, str(HERE))
from compare_fixedclock import cells  # noqa: E402

MARLIN = "#9a9993"
# graph-replay wall clock per layer call, us: (hot, cold) -> (Marlin, one kernel)
LAYER = {
    (9, 0): (103.6, 81.8), (9, 1): (120.5, 91.2), (11, 1): (137.8, 107.1),
    (7, 2): (128.3, 114.7), (9, 2): (128.8, 118.1), (12, 2): (157.6, 125.6),
    (0, 2): (119.1, 110.1), (9, 3): (172.8, 163.4),
}


def step_times():
    out = {"marlin": [], "tiered": []}
    for job in ("2085950", "2091191"):
        for f in sorted((HERE / f"e2e-ab-{job}").glob("decode-*.json")):
            arm = "marlin" if "marlin" in f.name else "tiered"
            if job == "2085950" and f.name == "decode-2-tiered.json":
                continue   # an earlier build (before the consumer fix)
            out[arm].append(json.loads(f.read_text())["median_itl_ms"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    prob = cells()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [2.4, 1]},
                                 facecolor=SURFACE)
    order = sorted(LAYER, key=lambda hc: (hc[1], hc[0]))
    xs = np.arange(len(order))
    w = 0.38
    m = [LAYER[hc][0] for hc in order]
    t = [LAYER[hc][1] for hc in order]
    ax.bar(xs - w / 2, m, w, color=MARLIN, edgecolor=SURFACE, linewidth=1.5, label="Marlin, two tiers on two streams")
    ax.bar(xs + w / 2, t, w, color=HBM, edgecolor=SURFACE, linewidth=1.5, label="one kernel for both tiers")
    for x, a, b in zip(xs, m, t):
        ax.text(x + w / 2, b + 3, f"{a / b:.2f}x", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    labels = []
    for h, c in order:
        p = prob.get((h, c))
        labels.append(f"{h} hot\n{c} cold\n{p:.1%}" if p else f"{h} hot\n{c} cold\n")
    ax.set_xticks(xs, labels, fontsize=9)
    ax.set_ylabel("one MoE layer, w13 + w2 (µs)")
    ax.set_ylim(0, 222)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    ax.set_title("Per layer, by experts this GPU runs (and how often that mix occurs)", fontsize=11,
                 color=INK, loc="left")
    ax.text(0.99, 0.97, "hot-heavy: 1.25-1.32x\ncold-bound (C2C limit): 1.06-1.12x",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK2)

    st = step_times()
    for i, (arm, color) in enumerate((("marlin", MARLIN), ("tiered", HBM))):
        ys = st[arm]
        bx.scatter(np.full(len(ys), i) + np.linspace(-0.08, 0.08, len(ys)), ys, s=60, color=color,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        mean = float(np.mean(ys))
        bx.plot([i - 0.25, i + 0.25], [mean, mean], color=INK, linewidth=1.5)
        bx.text(i + 0.3, mean, f"{mean:.2f} ms", va="center", fontsize=10, color=INK)
    bx.set_xticks([0, 1], ["Marlin", "one kernel"])
    bx.set_xlim(-0.5, 1.9)
    bx.set_ylim(17.5, 21)
    bx.set_ylabel("median decode step (ms)")
    bx.grid(axis="x", visible=False)
    bx.grid(axis="y", color=GRID, linewidth=0.8)
    gain = np.mean(st["marlin"]) - np.mean(st["tiered"])
    bx.set_title(f"End to end: -{gain:.2f} ms per step (-{gain / np.mean(st['marlin']):.0%})", fontsize=11,
                 color=INK, loc="left")
    bx.text(0.02, 0.03, "each dot: one server run, 2 nodes\nGSM8K-400: Marlin 0.880-0.900, one kernel 0.895-0.900",
            transform=bx.transAxes, fontsize=8.5, color=MUTED)

    fig.suptitle("MiMo-V2.6 decode MoE: one TMA kernel streaming HBM and Grace experts together vs Marlin",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
