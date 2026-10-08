#!/usr/bin/env python3
"""Figures for the 2026-10-04..08 decode work in the write-up (glm/README.md,
sections 8-9). Numbers are from the experiment READMEs named per figure.

    plot_micro.py <out dir>

glm-hot-experts.png     hot experts per GPU across the memory changes
glm-decode-changes.png  paired step-time deltas (95% CI), agentic and 50-130K
glm-decode-gemm.png     8-token dense GEMMs: HBM floor, cuBLAS, decode_gemm
glm-dcp-width.png       FlashMLA sparse decode vs index width; live slots seen
glm-step-now.png        where the decode step goes now (agentic, long context)
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26-mimo-routing-profile"))
from plot_routing import DDR, GRID, HBM, INK, INK2, MUTED, SURFACE  # noqa: E402

IDLE = "#dcdbd4"
Z = 1.96  # compare_ba.py prints a standard error


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def finish(fig, title, subtitle, path, top=0.84):
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, top + 0.035, subtitle, fontsize=9.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def hot_experts(out: Path):
    # 2026-10-06-skip-layer-kv-grace, 2026-10-07-mem-reclaim (startup logs)
    steps = [
        ("start\n(10-03)", 3180, None),
        ("skip-layer\nMLA KV\non Grace", 3358, "3.5 GiB"),
        ("fp8 drafter\nweights + KV", 3450, "1.8 GiB"),
        ("drafter KV\non Grace,\nreserve 4.7", 3531, "~1.6 GiB"),
        ("DCP workspaces,\none NCCL comm,\nembedding on Grace,\nreserve 1.7", 3676,
         "3.5 GiB, some\nof it reserve"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), facecolor=SURFACE)
    for i, (label, v, freed) in enumerate(steps):
        ax.bar(i, v, width=0.62, color=MUTED if i == 0 else HBM, edgecolor=SURFACE)
        ax.text(i, v + 12, f"{v:,}", ha="center", fontsize=10, color=INK, fontweight="bold")
        if i:
            d = v - steps[i - 1][1]
            ax.text(i, 3120, f"+{d}\n({freed})", ha="center", va="bottom", fontsize=9,
                    color="white")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=9, color=INK)
    ax.set_ylim(3100, 3720)
    ax.set_ylabel("hot experts per GPU (of 4,800)", color=INK2)
    style(ax)
    finish(fig, "Freed HBM goes to hot experts: +496 per GPU",
           "per GPU at startup (prod config, 400K pool); in brackets the HBM each change freed "
           "(20.3 MiB per INT4 expert).\nThe last also lowered a reserve that had been covering "
           "that memory.", out / "glm-hot-experts.png", top=0.8)


def changes(out: Path):
    # label, agentic (delta, se), long context (delta, se), README
    rows = [
        ("memory stack: skip-layer KV + fp8 drafter\n+ drafter KV on Grace (short runs)",
         (-0.54, 0.10), None),
        ("smaller workspaces, one NCCL comm,\nembedding on Grace; reserve 1.7 (+145 hot)",
         (-0.223, 0.030), (-0.265, 0.033)),
        ("hot set promoted by frequency,\nnot expert id", (-0.199, 0.025), (-0.324, 0.029)),
        ("8-token dense GEMMs on our kernel", (-0.472, 0.027), (-0.498, 0.036)),
        ("DFlash2 drafter as a CUDA graph", (-0.088, 0.024), (-0.072, 0.027)),
        ("DCP sparse attention: 768-wide\nindex rows instead of 2048", (-0.472, 0.027),
         (-0.478, 0.021)),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURFACE)
    for i, (label, ag, lc) in enumerate(rows):
        y = len(rows) - 1 - i
        for val, off, color in ((ag, 0.13, HBM), (lc, -0.13, DDR)):
            if val is None:
                continue
            d, se = val
            ax.plot([d - Z * se, d + Z * se], [y + off] * 2, color=color, linewidth=2,
                    solid_capstyle="round")
            ax.plot([d], [y + off], "o", color=color, markersize=7.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.text(d - Z * se - 0.02, y + off, f"{d:+.2f}", ha="right", va="center",
                    fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(-0.9, 0.05)
    ax.set_xlabel("change in decode step time, ms (paired, 95% CI)", color=INK2)
    handles = [plt.Line2D([], [], color=c, marker="o", linewidth=2) for c in (HBM, DDR)]
    ax.legend(handles, ["agentic tasks", "50K / 130K-token decode"], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5)
    style(ax, "x")
    ax.tick_params(axis="y", length=0)
    finish(fig, "Decode changes since the verify-step work: about -2 ms per step",
           "each against its own control, paired on the same nodes (4-6; the first row one "
           "node, short runs);\noutputs equal, bitwise or to bf16 rounding",
           out / "glm-decode-changes.png", top=0.8)


def decode_gemm(out: Path):
    # 2026-10-07-skinny-gemm-v2 (isolated, L2 flushed)
    shapes = ["o_proj\n48 MiB", "fused qkv_a\n31 MiB", "q_b\n16 MiB"]
    floor = [13.8, 8.9, 4.6]
    cublas = [22.4, 17.0, 8.9]
    ours = [18.0, 13.8, 8.5]
    x = np.arange(len(shapes))
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    for k, (vals, color, label) in enumerate(((floor, IDLE, "HBM floor (weights at 3.64 TB/s)"),
                                              (cublas, MUTED, "cuBLAS"),
                                              (ours, HBM, "decode_gemm"))):
        ax.bar(x + (k - 1) * 0.26, vals, 0.25, color=color, label=label, edgecolor=SURFACE)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 1) * 0.26, v + 0.3, f"{v:.1f}", ha="center", fontsize=9,
                    color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, fontsize=9.5, color=INK)
    ax.set_ylabel("us per call, 8 tokens", color=INK2)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    style(ax)
    finish(fig, "8-token GEMMs only stream weights: a kernel built for that",
           "bf16 weights, 8 activations; 4 warps per tile with a long K each and "
           "double-buffered 16-byte loads. 78 calls each per step.",
           out / "glm-decode-gemm.png")


def dcp_width(out: Path):
    # 2026-10-08-flashmla-split: bench-2226888.txt and live-slots-per-window.txt
    widths = [2048, 1024, 768, 640]
    us5k = [21.37, 16.99, 16.75, 15.68]
    us100k = [21.57, 17.44, 16.75, 15.44]
    live = []
    for line in (HERE.parent / "2026-10-08-flashmla-split" / "live-slots-per-window.txt"
                 ).read_text().splitlines():
        live.append(int(line.split()[1].rstrip(",")))
    live = np.array(live)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(widths))
    a1.bar(x - 0.2, us5k, 0.38, color=HBM, label="5K context")
    a1.bar(x + 0.2, us100k, 0.38, color=DDR, label="100K context")
    for xi, v in zip(x, us5k):
        a1.text(xi - 0.2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    for xi, v in zip(x, us100k):
        a1.text(xi + 0.2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    a1.set_xticks(x)
    a1.set_xticklabels([f"{w}" + (" (was)" if w == 2048 else "") for w in widths],
                       fontsize=9.5, color=INK)
    a1.set_xlabel("index row width per GPU", color=INK2)
    a1.set_ylabel("us per layer (main + combine), 78 layers", color=INK2)
    a1.set_title("FlashMLA does the -1 slots too", loc="left", fontsize=10.5, color=INK)
    a1.legend(loc="upper right", frameon=False, fontsize=9)
    style(a1)
    bins = np.arange(200, 800, 10)
    a2.hist(live, bins=bins, color=HBM)
    for w, label, dx, ha in ((768, "width 768", -6, "right"), (512, "2048 / 4", -6, "right")):
        a2.axvline(w, color=DDR if w == 768 else INK2, linestyle="-" if w == 768 else "--",
                   linewidth=1.5)
        a2.text(w + dx, a2.get_ylim()[1] * 0.92, label, color=INK2, fontsize=9, ha=ha)
    a2.set_xlabel("most live slots in any row, per 1,000-step window", color=INK2)
    a2.set_ylabel("windows", color=INK2)
    a2.set_title(f"Served: max {live.max()}, 0 rows over 768", loc="left", fontsize=10.5,
                 color=INK)
    style(a2)
    finish(fig, "Under DCP4 a GPU owns ~1/4 of the 2,048 picked tokens",
           "left: microbench at the served shape (8 tokens x 64 heads); right: 248 "
           "windows of the 768-width A/B (short GSM8K prompts form the left cluster)",
           out / "glm-dcp-width.png", top=0.82)


def parse_breakdown(text: str, windows: list[str]) -> dict:
    parts: dict = {}
    for block in re.split(r"^== ", text, flags=re.M)[1:]:
        name = block.split()[0]
        if name not in windows:
            continue
        for m in re.finditer(r"^\s+([0-9.]+) ms\s+[0-9.]+%\s+(.+)$", block, flags=re.M):
            parts.setdefault(m.group(2).strip(), []).append(float(m.group(1)))
    return {k: float(np.mean(v)) for k, v in parts.items()}


def step_now(out: Path):
    text = (HERE / "breakdown-dd3-w768-rank0.txt").read_text()
    arms = [("agentic tasks", parse_breakdown(text, ["window-0", "window-1", "window-2"])),
            ("50K / 130K decode", parse_breakdown(text, ["window-10", "window-11"]))]
    groups = [  # display name, color, buckets
        ("MoE", HBM,
         ["MoE GEMMs (tiered one-kernel w13/w2)", "MoE route/act/finalize + router"]),
        ("all-reduce + norm (mostly waiting on the MoE)", "#7fb0e8",
         ["all-reduce + RMSNorm (fused, incl. wait)"]),
        ("dense GEMMs", DDR, ["dense GEMMs (decode_gemm)",
                             "dense GEMMs (cuBLAS / CUTLASS / cute-dsl)"]),
        ("attention, indexer, DCP, KV writes", "#f2a37f",
         ["attention (FlashMLA main)", "attention (split-KV combine)", "DSA indexer",
          "DCP collectives (gather / LSE reduce-scatter)", "KV write / skip-KV staging"]),
        ("drafter, sampling, other", MUTED, ["drafter / sampling / other"]),
        ("idle", IDLE, ["GPU idle"]),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.4), facecolor=SURFACE)
    for j, (name, parts) in enumerate(arms):
        y = 1 - j
        left = 0.0
        for gname, color, keys in groups:
            v = sum(parts.get(k, 0.0) for k in keys)
            ax.barh(y, v, left=left, height=0.55, color=color, edgecolor=SURFACE,
                    linewidth=2, label=gname if j == 0 else None)
            if v > 0.8:
                ax.text(left + v / 2, y, f"{v:.1f}", ha="center", va="center", fontsize=9.5,
                        color=INK if color in (IDLE, "#f2a37f", "#7fb0e8") else "white")
            left += v
        ax.text(left + 0.15, y, f"{left:.1f} ms", va="center", fontsize=10, color=INK,
                fontweight="bold")
    ax.set_yticks([1, 0])
    ax.set_yticklabels([a[0] for a in arms], fontsize=9.5, color=INK)
    ax.set_xlim(0, 23.5)
    ax.set_xlabel("ms per decode step (rank 0, each instant counted once)", color=INK2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.8)
    style(ax, "x")
    ax.tick_params(axis="y", length=0)
    finish(fig, "Where a decode step goes now",
           "prod config with all the changes, 8-token verify + DFlash2 draft; "
           "torch-profiler windows", out / "glm-step-now.png", top=0.72)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for f in (hot_experts, changes, decode_gemm, dcp_width, step_now):
        f(out)


if __name__ == "__main__":
    main()
