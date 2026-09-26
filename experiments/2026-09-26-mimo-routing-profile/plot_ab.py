#!/usr/bin/env python3
"""Dot plot of the placement A/B: decode step time per run, linear vs profile.

    plot_ab.py --out figs/5-decode-ab.png ab-<job>...
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_routing import DDR, HBM, INK, INK2, SURFACE  # noqa: F401  (sets rcParams)
from summarize_ab import tokens_per_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args()
    runs = {"linear": [], "profile": []}
    for directory in args.dirs:
        for decode in directory.glob("decode-*.json"):
            tag = decode.stem.removeprefix("decode-")
            arm = tag.split("-", 1)[1]
            tpot = json.loads(decode.read_text())["mean_tpot_ms"]
            runs[arm].append(tpot * tokens_per_step(directory / f"spec-{tag}.txt"))

    fig, ax = plt.subplots(figsize=(8, 2.8))
    rows = (("linear", "today: arbitrary\nhot set", DDR), ("profile", "routing profile\nhot set", HBM))
    rng = np.random.default_rng(0)
    for y, (arm, label, color) in zip((1, 0), rows):
        values = np.array(runs[arm])
        ax.plot(values, y + rng.uniform(-0.08, 0.08, len(values)), "o", color=color,
                markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(values.mean(), y + 0.28, f"{values.mean():.1f} ms  (n={len(values)})",
                ha="center", color=INK, fontsize=11)
    linear, profile = np.mean(runs["linear"]), np.mean(runs["profile"])
    ax.set_yticks([1, 0], [rows[0][1], rows[1][1]])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlim(20, 38)
    ax.set_xlabel("decode step time, ms (8-token DFlash verify, lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"MiMo-V2.6 decode: {(1 - profile / linear) * 100:.0f}% faster steps "
                 "from placement alone", pad=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print({arm: [round(v, 2) for v in values] for arm, values in runs.items()})


if __name__ == "__main__":
    main()
