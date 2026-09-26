#!/usr/bin/env python3
"""Figures for the replica section of the gist.

    plot_replicas.py --trace-dir DIR --profile profile-3827-r1500.json --out gist/

replicas.png      one real held-out layer: each GPU's active cold experts
                  without copies and with the runtime's min-max assignment
mimo-ladder.png   MiMo decode step time across the day's changes (measured)
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from mimo_replicas import EP, active_mask, load_steps
from plot_routing import DDR, HBM, INK, INK2, SURFACE

NEUTRAL = "#c3c2b7"


def assign(experts, owners_row, secondary_row):
    """Exact min-max assignment of one layer's active cold experts."""
    fixed = [e for e in experts if secondary_row[e] < 0]
    flexible = [e for e in experts if secondary_row[e] >= 0]
    best = None
    for choice in itertools.product((0, 1), repeat=len(flexible)):
        load = np.zeros(EP, dtype=int)
        where = {}
        for e in fixed:
            where[e] = owners_row[e]
        for e, pick in zip(flexible, choice):
            where[e] = secondary_row[e] if pick else owners_row[e]
        for rank in where.values():
            load[rank] += 1
        key = (load.max(), sum(choice))  # fewest moves among the optima
        if best is None or key < best[0]:
            best = (key, where)
    return best[1]


def pick_case(active, owners, hot, secondary):
    """A held-out step/layer where copies lower the busiest GPU by the typical 2."""
    cold = active & ~hot[None]
    for step in range(0, active.shape[0], 97):
        for layer in range(active.shape[1]):
            experts = np.flatnonzero(cold[step, layer])
            if not 8 <= len(experts) <= 11:
                continue
            flexible = [e for e in experts if secondary[layer, e] >= 0]
            if len(flexible) > 12:
                continue
            before = np.bincount(owners[layer, experts], minlength=EP)
            where = assign(experts, owners[layer], secondary[layer])
            after = np.bincount(list(where.values()), minlength=EP)
            if before.max() - after.max() == 2 and before.max() >= 5:
                return step, layer, experts, where
    raise SystemExit("no representative case found")


def fig_replicas(case, owners, out: Path) -> None:
    step, layer, experts, where = case
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    panels = (("without copies", {e: owners[layer, e] for e in experts}),
              ("with copies (what the runtime picks)", where))
    for ax, (title, placement) in zip(axes, panels):
        for rank in range(EP):
            mine = [e for e in experts if placement[e] == rank]
            for height, e in enumerate(sorted(mine, key=lambda e: placement[e] != owners[layer, e])):
                moved = owners[layer, e] != rank
                ax.bar(rank, 0.9, bottom=height + 0.05, width=0.62,
                       color=SURFACE if moved else DDR, edgecolor=DDR,
                       hatch="///" if moved else None, linewidth=1.5)
                ax.text(rank, height + 0.5, f"#{e}", ha="center", va="center", fontsize=8,
                        color=INK if moved else "white")
        loads = np.bincount(list(placement.values()), minlength=EP)
        ax.axhline(loads.max(), color=INK, linewidth=1.2, linestyle="--")
        ax.text(3.45, loads.max() + 0.08, f"layer waits for\n{loads.max()} cold experts",
                ha="right", va="bottom", color=INK, fontsize=9)
        ax.set_xticks(range(EP), [f"GPU {r}" for r in range(EP)])
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.6, 3.6)
    axes[0].set_ylabel("cold (Grace) experts this GPU runs")
    axes[0].set_ylim(0, 7.2)
    fig.legend(handles=[Patch(color=DDR, label="the GPU's own cold expert"),
                        Patch(facecolor=SURFACE, edgecolor=DDR, hatch="///",
                              label="a copy it holds of another GPU's expert")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(f"One real MoE layer (layer {layer + 1}, a held-out decode step): "
                 "copies let the busiest GPU hand off work",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "replicas.png", dpi=160)
    plt.close(fig)


def fig_ladder(out: Path) -> None:
    steps = [
        ("arbitrary\nhot set", 34.73, True),
        ("hot set from\nrouting traces", 25.03, True),
        ("no sequence-\nparallel MoE", 21.82, False),
        ("+ replicas", 20.73, True),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(steps))
    ax.bar(x, [v for _, v, _ in steps], width=0.6, edgecolor=SURFACE, linewidth=2,
           color=[HBM if offload else NEUTRAL for *_, offload in steps])
    for xi, (_, value, _) in zip(x, steps):
        ax.text(xi, value + 0.5, f"{value:.1f} ms", ha="center", va="bottom", color=INK)
        if xi:
            ax.text(xi, value / 2, f"{(value / steps[0][1] - 1) * 100:.0f}%", ha="center",
                    va="center", color="white" if steps[xi][2] else INK, fontsize=11,
                    fontweight="bold")
    ax.set_xticks(x, [name for name, *_ in steps])
    ax.set_ylabel("decode step, ms (8 tokens verified)")
    ax.set_ylim(0, 42)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=HBM, label="offload work"),
                       Patch(color=NEUTRAL, label="everything else")], loc="upper right")
    ax.set_title("MiMo-V2.6 decode on one 4x GH200 node: 110 -> 178 tok/s\n(labels: change vs the first bar)",
                 pad=10)
    fig.tight_layout()
    fig.savefig(out / "mimo-ladder.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    owners = np.asarray(profile["owners"])
    secondary = np.asarray(profile["secondary_ranks"])
    hot = np.zeros(owners.shape, dtype=bool)
    for layer, ids in enumerate(profile["hot_experts"]):
        hot[layer, ids] = True
    held = active_mask(load_steps(args.trace_dir, "heldout", profile["routed_layers"]),
                       owners.shape[1])
    case = pick_case(held, owners, hot, secondary)
    print("case: step", case[0], "layer", case[1] + 1, "experts", list(case[2]))
    fig_replicas(case, owners, args.out)
    fig_ladder(args.out)


if __name__ == "__main__":
    main()
