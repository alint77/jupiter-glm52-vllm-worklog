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
    ("MoE routing + collectives", ["MoE routing/align/sum/act", "MoE all-gather (SP)",
                                   "MoE reduce-scatter (SP)", "TP all-reduce"]),
    ("dense GEMMs", ["dense GEMM"]),
    ("attention", ["attention"]),
    ("other", ["norm/rope/elementwise"]),
)


def step_groups(result: dict) -> dict[str, float]:
    """Mean-over-GPUs ms per step by group; 'other' absorbs drafter, logits,
    host kernels and idle so the groups sum to the step period.

    Waiting is each collective's mean-over-ranks minus its cross-rank minimum,
    summed over collective kinds, and is taken out of the collectives group.
    """
    ranks = result["ranks"].values()
    cat = lambda name: statistics.fmean(r["target_categories_ms"].get(name, 0) for r in ranks)  # noqa: E731
    waiting = sum(c["mean_ms_per_step"] - c["cross_rank_min_ms_per_step"]
                  for c in result["collectives"].values())
    values = {label: sum(cat(n) for n in names) for label, names in GROUPS}
    values["waiting for slowest GPU"] = waiting
    values["MoE routing + collectives"] -= waiting
    period = statistics.fmean(r["period_ms"]["mean"] for r in ranks)
    values["other"] += period - sum(values.values())
    return values


def fig_steps(rows: list[tuple[str, dict]], out: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 1.2 + 1.1 * len(rows)))
    ys = list(range(len(rows)))[::-1]
    for y, (label, result) in zip(ys, rows):
        left = 0.0
        groups = step_groups(result)
        for (group, _), color in zip(GROUPS, SLOTS):
            width = groups[group]
            ax.barh(y, width, left=left, color=color, height=0.55, edgecolor=SURFACE,
                    linewidth=2, label=group if y == ys[0] else None)
            if width > 1.2:
                ax.text(left + width / 2, y, f"{width:.1f}", ha="center", va="center",
                        color="white" if color in (SLOTS[0], SLOTS[5], SLOTS[6]) else INK,
                        fontsize=9)
            left += width
        ax.text(left + 0.4, y, f"{left:.1f} ms", va="center", color=INK, fontsize=10)
    ax.set_yticks(ys, [label for label, _ in rows])
    ax.set_xlim(0, 41)
    ax.set_xlabel("one decode step (8-token DFlash verify), ms, mean of 4 GPUs")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45 + 0.08 * len(rows)), ncol=4,
              fontsize=9)
    ax.set_title("MiMo-V2.6 decode step, by where the time goes", pad=10)
    fig.tight_layout()
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def moe_layer(ops: list[dict], k: int):
    """(segment start, segment ops, closing collective) of the k-th MoE layer.

    A collective closes a MoE layer when Marlin ran since the previous one: the
    reduce-scatter under sequence-parallel MoE, the MoE all-reduce without it.
    """
    left, found = ops[0]["t"], 0
    for c in (o for o in ops if "ReduceScatter" in o["name"]
              or "cross_device_reduce" in o["name"]):
        seg = [o for o in ops if left <= o["t"] < c["t"]]
        if any("marlin_moe" in o["name"] for o in seg):
            if found == k:
                return left, seg, c
            found += 1
        left = c["end"]
    raise IndexError(k)


def fig_layer(trace_dir: Path, out: Path, name: str) -> None:
    per_rank = {}
    for path in sorted(trace_dir.glob("*.pt.trace.json.gz")):
        per_rank[int(RANK_RE.search(path.name).group(1))] = steps(load(path))
    step = 20
    ranked = []
    for k in range(69):
        layer = {}
        for rank, rows in per_rank.items():
            ops = sorted(rows[step]["phases"]["target"], key=lambda e: e["t"])
            layer[rank] = moe_layer(ops, k)
        waits = [(v[2]["end"] - v[2]["t"]) for v in layer.values()]
        ranked.append((max(waits) - min(waits), k, layer))
    ranked.sort()
    spread, k, layer = ranked[len(ranked) * 3 // 4]  # an upper-quartile layer, not the worst
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for y, rank in enumerate(sorted(layer, reverse=True)):
        left, seg, rs = layer[rank]
        origin = rs["end"]  # the collective completes together on all ranks
        for op in seg:
            kind = category(op)
            color = {"MoE hot Marlin (HBM)": SLOTS[0], "MoE cold Marlin (Grace)": SLOTS[1]}.get(kind, "#c3c2b7")
            lane = y + (0.18 if kind.startswith("MoE hot") else -0.18 if kind.startswith("MoE cold") else 0)
            height = 0.3 if "Marlin" in kind else 0.7
            ax.barh(lane, (op["end"] - op["t"]) * 1000, left=(op["t"] - origin) * 1000,
                    color=color, height=height, linewidth=0)
        ax.barh(y, (rs["end"] - rs["t"]) * 1000, left=(rs["t"] - origin) * 1000,
                color=SLOTS[2], height=0.7, linewidth=0)
    ax.set_yticks(range(4), [f"GPU {r}" for r in sorted(layer, reverse=True)])
    ax.set_xlabel("µs before the layer's MoE collective completes (together on all GPUs)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SLOTS[0], label="hot Marlin (HBM)"),
                       Patch(color=SLOTS[1], label="cold Marlin (Grace)"),
                       Patch(color=SLOTS[2], label="MoE collective = waiting for slowest GPU"),
                       Patch(color="#c3c2b7", label="everything else in the layer")],
              loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"One MoE layer (layer {k + 1}, step {step}) on the four GPUs: "
                 "the GPU with more cold experts sets the pace", pad=10)
    fig.tight_layout()
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-trace", type=Path, required=True,
                        help="decode trace dir for the per-GPU layer figure")
    parser.add_argument("--row", nargs=3, action="append", required=True,
                        metavar=("LABEL", "BREAKDOWN_JSON", "KEY"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [(label, json.loads(Path(path).read_text())[key]) for label, path, key in args.row]
    fig_steps(rows, args.out, f"step-breakdown{args.suffix}.png")
    fig_layer(args.layer_trace, args.out, f"layer-ranks{args.suffix}.png")
    for label, result in rows:
        print(label, {k: round(v, 2) for k, v in step_groups(result).items()})


if __name__ == "__main__":
    main()
