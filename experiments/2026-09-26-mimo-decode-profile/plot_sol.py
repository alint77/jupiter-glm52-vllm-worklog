#!/usr/bin/env python3
"""Figures for the decode SOL analysis.

    plot_sol.py --sol sol-2077518.json --count marlin-count-short-2077518.json --out figs/

sol-step.png         per component: measured ms/step against its floor
marlin-by-count.png  Marlin w13+w2 time and roof fraction vs active experts
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-routing-profile"))
from plot_routing import DDR, HBM, INK, INK2, MUTED, SURFACE  # noqa: E402

EXPERT_BYTES = 20_054_016
ROOF = {"hot": 3.626e12, "cold": 0.409e12}
NEUTRAL = "#c3c2b7"


def marlin_sol_ms(count: dict, tier: str) -> float:
    rows = count["tiers"][tier]
    samples = sum(v["samples"] for v in rows.values())
    experts = sum(int(n) * v["samples"] for n, v in rows.items())
    return experts / (samples / 69) * EXPERT_BYTES / ROOF[tier] * 1e3


def fig_step(sol: dict, count: dict, out: Path) -> None:
    rows = [
        ("MoE Marlin hot (HBM)", sol["MoE Marlin hot"]["ms_per_step"], marlin_sol_ms(count, "hot"), "bw"),
        ("MoE Marlin cold (Grace)", sol["MoE Marlin cold"]["ms_per_step"], marlin_sol_ms(count, "cold"), "bw"),
        ("TP all-reduce (141)", sol["TP all-reduce"]["ms_per_step"], 0.61, "lat"),
        ("MoE sum / act / topk (345)", sol["MoE sum / act / topk"]["ms_per_step"], None, "lat"),
        ("o_proj, bf16", sol["o_proj (bf16)"]["ms_per_step"], sol["o_proj (bf16)"]["sol_ms_per_step"], "bw"),
        ("qkv_proj, fp8", sol["qkv_proj (fp8)"]["ms_per_step"], sol["qkv_proj (fp8)"]["sol_ms_per_step"], "bw"),
        ("norms / rope / elementwise (527)", sol["norms / rope / elementwise"]["ms_per_step"], None, "lat"),
        ("replica assign (69)", sol["replica assign + align"]["ms_per_step"], None, "lat"),
        ("DFlash drafter", sol["DFlash drafter (all but lm_head)"]["ms_per_step"], None, "lat"),
        ("attention, sliding 128 (60)", sol["attention, sliding (FA4)"]["ms_per_step"], None, "lat"),
        ("lm_head x3, bf16", sol["lm_head (bf16)"]["ms_per_step"], sol["lm_head (bf16)"]["sol_ms_per_step"], "bw"),
        ("router gate, bf16", sol["router gate (bf16)"]["ms_per_step"], sol["router gate (bf16)"]["sol_ms_per_step"], "bw"),
        ("attention, full (10)", sol["attention, full (FA3)"]["ms_per_step"], sol["attention, full (FA3)"]["sol_ms_per_step"], "bw"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(rows))[::-1]
    for yi, (name, measured, floor, kind) in zip(y, rows):
        ax.barh(yi, measured, height=0.6, color=NEUTRAL if kind == "lat" else HBM,
                edgecolor=SURFACE, linewidth=1)
        if floor is not None:
            ax.plot([floor, floor], [yi - 0.38, yi + 0.38], color=INK, linewidth=2.2)
        label = f"{measured:.2f} ms"
        if floor is not None and kind == "bw":
            label += f"  ({floor / measured:.0%} of SOL)"
        elif kind == "lat":
            label += "  latency-bound" if floor is None else f"  (floor {floor:.2f})"
        ax.text(measured + 0.08, yi, label, va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("ms per decode step (sum of kernel time, mean of 4 GPUs); hot and cold Marlin overlap")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=HBM, label="bandwidth-bound: measured"),
                       Line2D([], [], color=INK, linewidth=2.2, label="speed-of-light (bytes / roof)"),
                       Patch(color=NEUTRAL, label="small kernels: launch/latency-bound")],
              loc="lower right", fontsize=9)
    ax.set_title("MiMo decode step (21 ms, chat, short context): each part vs its floor",
                 pad=10)
    fig.tight_layout()
    fig.savefig(out / "sol-step.png", dpi=160)
    plt.close(fig)


def fig_count(count: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, tier, color, roof_name in ((axes[0], "hot", HBM, "HBM 3.63 TB/s"),
                                       (axes[1], "cold", DDR, "C2C 0.41 TB/s")):
        rows = {int(n): v for n, v in count["tiers"][tier].items() if int(n) > 0 and v["samples"] >= 20}
        n = np.array(sorted(rows))
        med = np.array([rows[k]["total_us"] for k in n])
        lo = np.array([rows[k]["total_p10"] for k in n])
        hi = np.array([rows[k]["total_p90"] for k in n])
        ax.fill_between(n, lo, hi, color=color, alpha=0.18, linewidth=0, label="10th-90th percentile")
        ax.plot(n, med, "o-", color=color, markersize=5, linewidth=2, label="measured median")
        ax.plot(n, n * EXPERT_BYTES / ROOF[tier] * 1e6, "--", color=INK, linewidth=1.5,
                label=f"speed of light ({roof_name})")
        fit = count["tiers"][tier + "_fit"]
        ax.text(0.03, 0.97, f"{fit['fixed_us']:.0f} µs + {fit['per_expert_us']:.1f} µs per expert\n"
                f"marginal {fit['marginal_gbps']:.0f} GB/s "
                f"({fit['marginal_gbps'] * 1e9 / ROOF[tier]:.0%} of roof)",
                transform=ax.transAxes, va="top", fontsize=10, color=INK)
        for k in (n[0], n[len(n) // 2], n[-1]):
            ax.annotate(f"{rows[k]['roof_fraction']:.0%}", (k, rows[k]["total_us"]),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=INK2)
        ax.set_xlabel(f"{tier} experts this GPU runs in the layer")
        ax.set_ylabel("w13 + w2 Marlin time (µs)")
        ax.set_title(f"{tier} tier ({'HBM' if tier == 'hot' else 'Grace over C2C'})", fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0, None)
    fig.suptitle("Routed Marlin vs active experts (labels: % of roof), 8-token verify, production overlap",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / "marlin-by-count.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sol", type=Path, required=True)
    parser.add_argument("--count", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sol = json.loads(args.sol.read_text())["decode-short"]
    count = json.loads(args.count.read_text())
    fig_step(sol, count, args.out)
    fig_count(count, args.out)


if __name__ == "__main__":
    main()
