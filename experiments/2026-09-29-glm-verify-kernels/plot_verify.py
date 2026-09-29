#!/usr/bin/env python3
"""Figures for the verify-step kernel work (numbers from README.md here).

    plot_verify.py <out dir>

glm-verify-changes.png  paired task-set step-time deltas with 95% CIs
glm-allreduce.png       all-reduce cost per step: work vs waiting, HEAD vs fused
glm-imbalance.png       busiest-GPU cold reads by holder scheme; hot imbalance
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-routing-profile"))
from plot_routing import DDR, GRID, HBM, INK, INK2, MUTED, SURFACE  # noqa: E402

IDLE = "#dcdbd4"


def finish(fig, title, subtitle, path, top=0.84):
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, top + 0.035, subtitle, fontsize=9.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def changes(out: Path):
    rows = [  # label, delta, lo, hi, kept
        ("Fused DCP4 query gather + attention combine", -0.22, -0.45, -0.03, True),
        ("Single-tile index remap; LSE read in place", -0.65, -0.87, -0.42, True),
        ("Reuse converted indices; MoE drops padding", -0.79, -0.93, -0.64, True),
        ("All three, against the start", -1.76, -2.00, -1.50, True),
        ("All-reduce + RMSNorm fusion (not enabled)", -0.15, -0.36, 0.00, False),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (label, d, lo, hi, kept) in enumerate(rows):
        y = len(rows) - 1 - i
        color = HBM if kept else MUTED
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot([d], [y], "o", color=color, markersize=9 if i == 3 else 8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(lo - 0.04, y, f"{d:+.2f} ms", ha="right", va="center", fontsize=9.5,
                color=INK, fontweight="bold" if i == 3 else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.axhline(1.5, color=GRID, linewidth=1)
    ax.set_xlim(-2.45, 0.25)
    ax.set_xlabel("change in decode step time, ms (paired, 95% CI)", color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    finish(fig, "Verify-step kernel changes: 26.0 -> 24.3 ms per step",
           "agentic task set; each change against its own control on the same requests. "
           "Outputs identical except the fusion.", out / "glm-verify-changes.png")


def allreduce(out: Path):
    arms = ["HEAD\n(custom all-reduce\n+ 2 norm kernels)", "fused\n(FlashInfer trtllm\nall-reduce + norm)"]
    segs = [  # (label, color, [HEAD, fused])
        ("all-reduce transfer", HBM, [0.67, 0.87]),
        ("add + RMSNorm kernels", DDR, [0.47, 0.0]),
        ("waiting for the slowest GPU", IDLE, [2.26, 1.95]),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for j, arm in enumerate(arms):
        y = 1 - j
        left = 0.0
        for label, color, vals in segs:
            v = vals[j]
            if v <= 0:
                continue
            ax.barh(y, v, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2)
            ax.text(left + v / 2, y, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if color != IDLE else INK)
            left += v
    ax.set_yticks([1, 0])
    ax.set_yticklabels(arms, fontsize=9.5, color=INK)
    ax.set_xlabel("ms per decode step, summed over ~150 all-reduces", color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in segs]
    ax.legend(handles, [s[0] for s in segs], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3, frameon=False, fontsize=9.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    finish(fig, "The all-reduce is mostly waiting, not transfer",
           "per step, fastest rank = work. Fusing the norm saves ~0.27 ms of work; "
           "FlashInfer's all-reduce is slower (5.5 vs 4.2 us).", out / "glm-allreduce.png",
           top=0.8)


def imbalance(out: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    cold = [("owner only", 204.0), ("owner + 2nd holder\n(served)", 150.8),
            ("any GPU\n(ceiling)", 141.0)]
    cold_mean = 112.6
    for i, (name, v) in enumerate(cold):
        a1.bar(i, v, width=0.6, color=MUTED if i == 0 else DDR, edgecolor=SURFACE, linewidth=2)
        a1.text(i, v + 4, f"{v:.0f}", ha="center", fontsize=10, color=INK)
    a1.axhline(cold_mean, color=INK2, linestyle="--", linewidth=1.2)
    a1.text(1.5, cold_mean + 3, f"mean GPU {cold_mean:.0f}", ha="center", va="bottom",
            fontsize=9, color=INK2)
    a1.set_xticks(range(len(cold)))
    a1.set_xticklabels([c[0] for c in cold], fontsize=9.5, color=INK)
    a1.set_ylabel("cold experts read on the busiest GPU\nper step (75 layers)", color=INK2)
    a1.set_title("Cold: more holders buy <= 10 per step", loc="left", fontsize=10.5, color=INK)
    hot = [("mean GPU", 663.3), ("busiest GPU", 867.7)]
    for i, (name, v) in enumerate(hot):
        a2.bar(i, v, width=0.6, color=MUTED if i == 0 else HBM, edgecolor=SURFACE, linewidth=2)
        a2.text(i, v + 12, f"{v:.0f}", ha="center", fontsize=10, color=INK)
    a2.set_xticks(range(len(hot)))
    a2.set_xticklabels([h[0] for h in hot], fontsize=9.5, color=INK)
    a2.set_ylabel("hot experts run per step (75 layers)", color=INK2)
    a2.set_title("Hot (pinned): ~205 extra on the busiest", loc="left", fontsize=10.5,
                 color=INK)
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    fig.suptitle("Where the MoE imbalance comes from", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.905, "captured agentic routes, 24,587 held-out 8-token steps, "
             "capped (looping) requests excluded", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "glm-imbalance.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    changes(out)
    allreduce(out)
    imbalance(out)
    print("wrote", sorted(p.name for p in out.glob("glm-*.png")))
