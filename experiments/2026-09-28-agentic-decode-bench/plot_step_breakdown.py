#!/usr/bin/env python3
""" matplotlib figure: where one GLM-5.3 decode step goes.

Two aligned horizontal bars on one axis: the full 27.6 ms step on top, and
the target/verify CUDA graph (89% of it) zoomed below, with a labeled key.
Buckets are recomputed from the trace with the shared analyzer
(union-partition inside the graph, so segments are disjoint and sum to the
busy time). Light surface, reference categorical palette (validated:
adjacent CVD dE 9.1 / normal 19.6), ~2px surface gaps between segments.

    plot_step_breakdown.py <trace-dir>... [--out figs/glm53-step-breakdown.png]
"""

import argparse
import collections
import importlib.util
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "analyze", Path(__file__).resolve().parent.parent
    / "2026-09-26-mimo-decode-profile/analyze.py")
analyze = importlib.util.module_from_spec(SPEC)
sys.modules["analyze"] = analyze
SPEC.loader.exec_module(analyze)

RANK_RE = re.compile(r"_rank(\d+)\.")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
IDLE = "#e4e3dc"

# reference categorical palette, slots 1..7 (adjacent-validated order)
C_STEP = {"verify": "#2a78d6", "drafter": "#898781",
          "logits_host": "#b5b4ad", "gap": IDLE}
C_GRAPH = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

MERGE = {
    "MoE one-kernel w13": "chain",
    "MoE one-kernel w2": "chain",
    "MoE one-kernel route/act/finalize": "chain",
    "dense GEMM": "dense",
    "TP all-reduce": "ar",
    "MoE reduce-scatter (SP)": "ar",
    "MoE all-gather (SP)": "ar",
    "attention": "attn",
    "DSA indexer": "attn",
    "DCP one-shot collectives": "dcp",
    "MoE routing/align/sum/act": "glue",
    "norm/rope/elementwise": "glue",
}


def compute(dirs: list[Path]) -> dict:
    rows = []
    for d in dirs:
        path = sorted(d.glob("*rank0*.pt.trace.json.gz"))[0]
        rows.extend(analyze.steps(analyze.load(path)))
    n = len(rows)
    buckets = collections.defaultdict(list)
    spans, busies, periods = [], [], []
    drafter, logits, host = [], [], []
    for row in rows:
        span = max(o["end"] for o in row["phases"]["target"]) - \
            min(o["t"] for o in row["phases"]["target"])
        busy = analyze.union(row["phases"]["target"])
        spans.append(span)
        busies.append(busy)
        periods.append(row["period"])
        per_row = collections.defaultdict(float)
        for name, value in analyze.partition(row["phases"]["target"]).items():
            per_row[MERGE.get(name, name)] += value
        for name, value in per_row.items():
            buckets[name].append(value)
        drafter.append(analyze.union(row["phases"].get("draft", [])))
        logits.append(analyze.union(row["phases"].get("logits", [])))
        host.append(analyze.union(row["phases"].get("host-side", [])))
    m = statistics.fmean
    graph = {k: m(v) for k, v in buckets.items()}
    graph["idle"] = m(spans) - m(busies)
    step = {
        "verify": m(spans),
        "drafter": m(drafter),
        "logits_host": m(logits) + m(host),
        "gap": m(periods) - m(spans) - m(drafter) - m(logits) - m(host),
    }
    return {"n": n, "period": m(periods), "step": step, "graph": graph,
            "graph_span": m(spans)}


def swatch(ax, x, y, color, size=0.05):
    ax.add_patch(plt.Rectangle((x, y), size, size, facecolor=color,
                               edgecolor="none", transform=ax.transAxes))


def draw(data: dict, out: Path) -> None:
    fig = plt.figure(figsize=(11.6, 4.7), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0.045, 0.16, 0.735, 0.68])
    ax.set_facecolor(SURFACE)
    key = fig.add_axes([0.795, 0.10, 0.20, 0.76])
    key.set_facecolor(SURFACE)
    key.axis("off")

    period = data["period"]
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.62, 1.72)
    ax.set_yticks([])
    ax.set_xticks(range(0, 31, 5))
    ax.set_xticklabels([f"{t}" for t in range(0, 31, 5)], color=MUTED,
                       fontsize=8)
    ax.tick_params(axis="x", length=3, color="#c3c2b7")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("ms per decode step", color=MUTED, fontsize=8.5, labelpad=6)

    # top bar: the whole step
    step_order = ["verify", "drafter", "logits_host", "gap"]
    y1, h = 0.92, 0.50
    left = 0.0
    for name in step_order:
        width = data["step"][name]
        ax.barh(y1, width, left=left, height=h, facecolor=C_STEP[name],
                edgecolor=SURFACE, linewidth=1.4)
        left += width
    xspan = data["step"]["verify"]
    ax.text(xspan / 2, y1, "target / verify graph", ha="center", va="center",
            color="#ffffff", fontsize=9.5, fontweight="bold")
    ax.text(xspan / 2, y1 - 0.17, f"{xspan:.1f} ms · {xspan / period * 100:.0f}%"
            f" of the step", ha="center", va="center", color="#ffffff",
            fontsize=7.5)
    ax.text(-0.25, y1, "decode step", ha="right", va="center", color=INK,
            fontsize=8.5, fontweight="bold")
    ax.text(-0.25, y1 - 0.17, f"{period:.2f} ms", ha="right", va="center",
            color=MUTED, fontsize=7.5)

    # zoom connectors from the verify segment down to the detail bar
    y2 = -0.10
    for x in (0.0, xspan):
        ax.plot([x, x], [y1 - h / 2 - 0.03, y2 + h / 2 + 0.03], color="#c3c2b7",
                linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
    ax.text(xspan / 2, y1 - h / 2 - 0.10, "zoomed below", ha="center",
            va="bottom", color=MUTED, fontsize=7.5)

    # bottom bar: the verify graph, union-partitioned (disjoint, sums to busy)
    graph_labels = [
        ("chain", "tiered MoE chain"),
        ("dense", "dense + shared GEMM"),
        ("ar", "TP all-reduce"),
        ("attn", "attention + DSA"),
        ("glue", "norms + glue"),
        ("dcp", "DCP one-shots"),
        ("idle", "graph idle"),
    ]
    left = 0.0
    inside_ink = {"chain": INK, "dense": INK, "ar": INK, "attn": INK,
                  "glue": "#ffffff", "dcp": "#ffffff"}
    for (name, label), color in zip(graph_labels, C_GRAPH + [IDLE]):
        width = data["graph"][name]
        ax.barh(y2, width, left=left, height=h, facecolor=color,
                edgecolor=SURFACE, linewidth=1.4)
        if width >= 2.2:
            ax.text(left + width / 2, y2 + 0.07, label, ha="center",
                    va="center", color=inside_ink.get(name, INK), fontsize=7.8,
                    fontweight="bold")
            ax.text(left + width / 2, y2 - 0.10, f"{width:.1f}", ha="center",
                    va="center", color=inside_ink.get(name, INK), fontsize=7.2)
        left += width
    ax.text(-0.25, y2, "verify graph", ha="right", va="center", color=INK,
            fontsize=8.5, fontweight="bold")
    ax.text(-0.25, y2 - 0.17, f"{data['graph_span']:.2f} ms", ha="right",
            va="center", color=MUTED, fontsize=7.5)

    # title + subtitle
    fig.text(0.045, 0.94, "GLM-5.3 decode step: where 27.6 ms goes",
             color=INK, fontsize=12, fontweight="bold")
    fig.text(0.045, 0.885, "c=1 · M=8 verify · DFlash2 (eager) · DCP4 · TP4/EP4"
             f" · 280 profiled steps · trace glm53-agentic-df2-prof-2108844"
             " (reserve 10 GB)", color=MUTED, fontsize=8)

    # key: swatch, name, value, note -- doubles as the table view
    key.text(0.0, 1.03, "the step", color=INK, fontsize=8.5,
             fontweight="bold", transform=key.transAxes)
    step_notes = {
        "verify": "the target/verify CUDA graph (detailed below)",
        "drafter": "DFlash2 drafter, eager",
        "logits_host": "lm_head + vocab all-gather + host-side kernels",
        "gap": "host between phases; step-entry spikes live in-graph",
    }
    y = 0.955
    for name in step_order:
        value = data["step"][name]
        swatch(key, 0.0, y - 0.028, C_STEP[name])
        key.text(0.07, y, {"verify": "target / verify graph",
                           "drafter": "drafter",
                           "logits_host": "logits + host kernels",
                           "gap": "inter-phase idle"}[name],
                 color=INK, fontsize=7.6, va="center", transform=key.transAxes)
        key.text(1.0, y, f"{value:5.2f}  ({value / period * 100:4.1f}%)",
                 color=INK, fontsize=7.6, va="center", ha="right",
                 family="monospace", transform=key.transAxes)
        key.text(0.07, y - 0.046, step_notes[name], color=MUTED, fontsize=6.6,
                 va="center", transform=key.transAxes)
        y -= 0.105
    y -= 0.045
    key.text(0.0, y, "target / verify graph", color=INK, fontsize=8.5,
             fontweight="bold", transform=key.transAxes)
    y -= 0.075
    graph_colors = {name: color for (name, _), color
                    in zip(graph_labels, C_GRAPH + [IDLE])}
    graph_notes = {
        "chain": "w13+w2 serial, cold experts over C2C; act/finalize"
                 " PDL-hidden; shared expert hides here too",
        "dense": "640+ cuBLAS launches, grid 2-4; ~3 ms is the bf16"
                 " weight-streaming floor",
        "ar": "x157 one-stage ARs: 0.7 ms wire + 2.2 ms waiting for the"
              " slowest rank (step-entry AR#0 is bursty: p50 22 us, p90 4.9 ms)",
        "attn": "78x sparse-MLA splitkv + combine; DSA indexer on 21 layers",
        "glue": "norms + routing fills/casts/cat micro-kernels",
        "dcp": "177 gathers + 78 reduce-scatters (c=1 pays DCP for capacity)",
        "idle": "sub-us kernel-boundary bubbles",
    }
    for name, label in graph_labels:
        value = data["graph"][name]
        graph_span = data["graph_span"]
        swatch(key, 0.0, y - 0.028, graph_colors[name])
        key.text(0.07, y, label, color=INK, fontsize=7.6, va="center",
                 transform=key.transAxes)
        key.text(1.0, y, f"{value:5.2f}  ({value / graph_span * 100:4.1f}%)",
                 color=INK, fontsize=7.6, va="center", ha="right",
                 family="monospace", transform=key.transAxes)
        key.text(0.07, y - 0.046, graph_notes[name], color=MUTED, fontsize=6.6,
                 va="top", wrap=True, transform=key.transAxes)
        y -= 0.105
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "figs"
                    / "glm53-step-breakdown.png")
    args = ap.parse_args()
    data = compute(args.dirs)
    print(f"{data['n']} steps, period {data['period']:.2f} ms")
    print("step:", {k: round(v, 3) for k, v in data["step"].items()})
    print("graph:", {k: round(v, 3) for k, v in data["graph"].items()})
    draw(data, args.out)


if __name__ == "__main__":
    main()
