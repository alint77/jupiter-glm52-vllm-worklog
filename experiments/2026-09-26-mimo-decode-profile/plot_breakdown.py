#!/usr/bin/env python3
"""Figures for the MiMo decode profile.

    plot_breakdown.py <trace-root> --out figs/

step-breakdown.png   one decode step, short and ~96K context, by category
layer-ranks.png      one real MoE layer on the four GPUs: hot, cold, waiting
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-routing-profile"))
from plot_routing import GRID, INK, INK2, SURFACE  # noqa: E402,F401  (rcParams)

from analyze import RANK_RE, category, load, steps  # noqa: E402

# Reference palette, fixed slot order (blue, orange, aqua, yellow, magenta,
# green, violet); hot = HBM blue and cold = Grace orange as in the gist.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
GROUPS = (
    ("MoE hot tier (HBM)", ["MoE hot Marlin (HBM)"]),
    ("MoE cold tier (Grace)", ["MoE cold Marlin (Grace)"]),
    ("waiting for slowest GPU", []),
    ("MoE routing + SP comm", ["MoE routing/align/sum/act", "MoE all-gather (SP)"]),
    ("dense GEMMs", ["dense GEMM"]),
    ("attention", ["attention"]),
    ("other", ["norm/rope/elementwise", "TP all-reduce"]),
)


def step_groups(result: dict) -> dict[str, float]:
    ranks = result["ranks"].values()
    cat = lambda name: statistics.fmean(r["target_categories_ms"].get(name, 0) for r in ranks)  # noqa: E731
    rs = result["collectives"]["ReduceScatter"]
    waiting = rs["mean_ms_per_step"] - rs["cross_rank_min_ms_per_step"]
    values = {label: sum(cat(n) for n in names) for label, names in GROUPS}
    values["waiting for slowest GPU"] = waiting
    values["MoE routing + SP comm"] += cat("MoE reduce-scatter (SP)") - waiting
    period = statistics.fmean(r["period_ms"]["mean"] for r in ranks)
    values["other"] += period - sum(values.values())  # drafter, logits, host, idle
    return values


def fig_steps(results: dict, out: Path) -> None:
    rows = [("short context", results["decode-short"]), ("~96K context", results["decode-96k"])]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for y, (label, result) in zip((1, 0), rows):
        left = 0.0
        for (name, _), color in zip(GROUPS, SLOTS):
            width = step_groups(result)[name]
            ax.barh(y, width, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=2,
                    label=name if y == 1 else None)
            if width > 1.2:
                ax.text(left + width / 2, y, f"{width:.1f}", ha="center", va="center",
                        color="white" if color in (SLOTS[0], SLOTS[5], SLOTS[6]) else INK, fontsize=9)
            left += width
        ax.text(left + 0.4, y, f"{left:.1f} ms", va="center", color=INK, fontsize=10)
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.set_xlim(0, 41)
    ax.set_xlabel("one decode step (8-token DFlash verify), ms, mean of 4 GPUs")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=9)
    ax.set_title("MiMo-V2.6 decode step: MoE is ~60% of the step once waiting is counted", pad=10)
    fig.tight_layout()
    fig.savefig(out / "step-breakdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_layer(trace_dir: Path, out: Path) -> None:
    per_rank = {}
    for path in sorted(trace_dir.glob("*.pt.trace.json.gz")):
        per_rank[int(RANK_RE.search(path.name).group(1))] = steps(load(path))
    step = 20
    ranked = []
    for k in range(69):
        layer = {}
        for rank, rows in per_rank.items():
            ops = sorted(rows[step]["phases"]["target"], key=lambda e: e["t"])
            rs = [o for o in ops if "ReduceScatter" in o["name"]]
            left = rs[k - 1]["end"] if k else ops[0]["t"]
            seg = [o for o in ops if left <= o["t"] < rs[k]["t"]]
            layer[rank] = (left, seg, rs[k])
        waits = [(v[2]["end"] - v[2]["t"]) for v in layer.values()]
        ranked.append((max(waits) - min(waits), k, layer))
    ranked.sort()
    spread, k, layer = ranked[len(ranked) * 3 // 4]  # an upper-quartile layer, not the worst
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for y, rank in enumerate(sorted(layer, reverse=True)):
        left, seg, rs = layer[rank]
        origin = rs["end"]  # reduce-scatter completes together on all ranks
        for op in seg:
            name = category(op)
            color = {"MoE hot Marlin (HBM)": SLOTS[0], "MoE cold Marlin (Grace)": SLOTS[1]}.get(name, "#c3c2b7")
            lane = y + (0.18 if name.startswith("MoE hot") else -0.18 if name.startswith("MoE cold") else 0)
            height = 0.3 if "Marlin" in name else 0.7
            ax.barh(lane, (op["end"] - op["t"]) * 1000, left=(op["t"] - origin) * 1000,
                    color=color, height=height, linewidth=0)
        ax.barh(y, (rs["end"] - rs["t"]) * 1000, left=(rs["t"] - origin) * 1000,
                color=SLOTS[2], height=0.7, linewidth=0)
    ax.set_yticks(range(4), [f"GPU {r}" for r in sorted(layer, reverse=True)])
    ax.set_xlabel("µs before the layer's reduce-scatter completes (it completes together on all GPUs)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SLOTS[0], label="hot Marlin (HBM)"),
                       Patch(color=SLOTS[1], label="cold Marlin (Grace)"),
                       Patch(color=SLOTS[2], label="reduce-scatter = waiting for slowest GPU"),
                       Patch(color="#c3c2b7", label="everything else in the layer")],
              loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"One MoE layer (layer {k + 1}, step {step}) on the four GPUs: "
                 "the GPU with more cold experts sets the pace", pad=10)
    fig.tight_layout()
    fig.savefig(out / "layer-ranks.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--breakdown", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.breakdown.read_text())
    fig_steps(results, args.out)
    fig_layer(args.trace_root / "decode-short", args.out)
    print({label: {k: round(v, 2) for k, v in step_groups(results[label]).items()} for label in results})


if __name__ == "__main__":
    main()
