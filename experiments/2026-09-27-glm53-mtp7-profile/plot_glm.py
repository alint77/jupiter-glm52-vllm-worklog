#!/usr/bin/env python3
"""Figures for the GLM-5.3 part of the gh200-tiered-moe write-up (glm/README.md).

    plot_glm.py --trace-dir .../snap-1535650-a --profile glm53-w4a16-2496.json \
        --out <gh200-tiered-moe>/glm/figs [--only NAME...]

skew          how skewed GLM-5.3's routing is (training split)
coverage      held-out routes served from HBM vs experts kept there
residency     active cold experts per GPU per step vs hot experts per GPU
replicas      one held-out layer's per-GPU cold load, and the replica budgets
ab            every greedy A/B run: step time vs accepted tokens, per arm
ladder        decode step across the changes, each against its own control
breakdown     one decode step this morning vs now (profiles 2096362, 2097699)
gemm          dense projections at M=8 against their HBM floor

Routing figures rank on the training split and measure on the held-out one.
Held-out 8-position windows stand in for MTP7's 8-token verify steps.
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
sys.path.insert(0, str(HERE.parent / "2026-09-26-mimo-routing-profile"))
from mimo_replicas import EP, active_mask, evaluate, layer_problems, load_steps, optimum  # noqa: E402
from plot_routing import DDR, GRID, HBM, INK, INK2, MUTED, SURFACE  # noqa: E402,F401

EXPERTS_PER_GPU = 256 * 75 // 4  # routed experts per GPU, all layers
COLD_US = 55.0  # one 20.3 MiB INT4 expert over C2C at ~380 GB/s
# (hot experts per GPU, label, label offset in points)
MARKS = ((2284, "start: reserve 10", (12, 10)), (2427, "reserve 7", (12, -30)),
         (3211, "DCP4", (10, 8)))


def routes(trace_dir: Path, layers: list[int], n: int, split: str) -> np.ndarray:
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    counts = np.zeros((len(layers), n))
    for record in manifest:
        if record["split"] != split:
            continue
        r = np.load(trace_dir / record["file"])[:, layers, :].astype(np.int64)
        for li in range(len(layers)):
            counts[li] += np.bincount(r[:, li].ravel(), minlength=n)
    return counts


def fig_skew(train: np.ndarray, out: Path) -> None:
    n = train.shape[1]
    share = np.sort(train / train.sum(1, keepdims=True), 1)[:, ::-1] * 100
    rank = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(rank, np.percentile(share, 10, 0), np.percentile(share, 90, 0),
                    color=HBM, alpha=0.18, linewidth=0, label="10th-90th percentile of layers")
    ax.plot(rank, np.median(share, 0), color=HBM, linewidth=2, label="median layer")
    ax.axhline(100 / n, color=MUTED, linewidth=1.5, linestyle="--")
    ax.text(n, 100 / n * 1.12, f"uniform (1/{n} = {100 / n:.2f}%)", color=INK2,
            ha="right", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_xlim(1, n)
    ax.set_xlabel("expert rank within its layer (1 = most used)")
    ax.set_ylabel("share of the layer's routes (%)")
    top = np.median(share[:, 0]) / (100 / n)
    half = np.median(share[:, : n // 2].sum(1))
    ax.set_title(f"GLM-5.3's expert routing is skewed\nbusiest expert ~{top:.0f}x "
                 f"a uniform share; the top half takes {half:.0f}% of routes",
                 pad=12, linespacing=1.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "glm-expert-skew.png", dpi=160)
    plt.close(fig)


def fig_coverage(train: np.ndarray, held: np.ndarray, out: Path) -> None:
    n = train.shape[1]
    order = np.argsort(-train, 1, kind="stable")
    ranked = np.take_along_axis(held, order, 1)
    cum = np.cumsum(ranked, 1) / ranked.sum(1, keepdims=True) * 100
    x = np.arange(1, n + 1) / n * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(x, np.percentile(cum, 10, 0), np.percentile(cum, 90, 0),
                    color=HBM, alpha=0.18, linewidth=0)
    ax.plot(x, np.median(cum, 0), color=HBM, linewidth=2,
            label="keep each layer's most-used experts")
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1.5, linestyle="--",
            label="keep arbitrary experts")
    for hot, name, offset in MARKS:
        frac = hot / EXPERTS_PER_GPU * 100
        value = float(np.median(cum[:, int(round(frac / 100 * n)) - 1]))
        ax.plot([frac], [value], "o", color=HBM, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=5)
        ax.annotate(f"{name}: {frac:.0f}% in HBM -> {value:.0f}% of routes",
                    (frac, value), xytext=offset, textcoords="offset points",
                    fontsize=9.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("experts kept in HBM, per layer (%)")
    ax.set_ylabel("held-out routes served from HBM (%)")
    ax.set_title("Where GLM-5.3's expert reads come from, by HBM budget", pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "glm-hbm-coverage.png", dpi=160)
    plt.close(fig)


def fig_residency(train_act: np.ndarray, held_act: np.ndarray, owners: np.ndarray,
                  out: Path) -> None:
    freq = train_act.mean(0)
    xs = np.arange(1800, 4001, 100)
    cold = []
    for per_gpu in xs:
        hot = np.zeros_like(freq, dtype=bool)
        for r in range(EP):
            mine = np.where(owners == r)
            order = np.argsort(-freq[mine])[:per_gpu]
            hot[mine[0][order], mine[1][order]] = True
        cold.append(float((held_act & ~hot[None]).sum((1, 2)).mean() / EP))
    cold = np.array(cold)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(xs, cold, color=DDR, linewidth=2.2)
    ax2 = ax.secondary_yaxis("right", functions=(lambda c: c * COLD_US / 1000,
                                                 lambda m: m * 1000 / COLD_US))
    ax2.set_ylabel(f"~ms per step over C2C ({COLD_US:.0f} us per expert)", color=INK2)
    for hot, name, offset in MARKS:
        c = float(np.interp(hot, xs, cold))
        ax.plot([hot], [c], "o", color=DDR, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=5)
        ax.annotate(f"{name}\n{hot} hot -> {c:.0f} cold", (hot, c), xytext=offset,
                    textcoords="offset points", fontsize=9.5, color=INK)
    ax.set_xlabel(f"hot experts per GPU (of {EXPERTS_PER_GPU})")
    ax.set_ylabel("active cold experts per GPU per step")
    ax.set_title("Every ~300 more hot experts per GPU saves ~2 ms of C2C reads\n"
                 "(held-out 8-token steps, hot set = each GPU's most-used experts)",
                 pad=12, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(out / "glm-residency.png", dpi=160)
    plt.close(fig)


def assign(loops: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per-GPU cold load after the min-max assignment, found exhaustively: how
    many of each GPU pair's shared experts go to its lower-numbered GPU."""
    import itertools

    pairs = [(a, b) for a in range(EP) for b in range(a + 1, EP)]
    best, best_load = None, None
    for split in itertools.product(*(range(int(c) + 1) for c in edges)):
        load = loops.astype(int).copy()
        for (a, b), c, lo in zip(pairs, edges, split):
            load[a] += lo
            load[b] += int(c) - lo
        key = (load.max(), load.std())
        if best is None or key < best:
            best, best_load = key, load
    return best_load


def fig_replicas(held_act: np.ndarray, owners: np.ndarray, hot: np.ndarray,
                 secondary: np.ndarray, budgets: dict, out: Path) -> None:
    loops0, edges0 = layer_problems(held_act, owners, hot, np.full(owners.shape, -1))
    loops1, edges1 = layer_problems(held_act, owners, hot, secondary)
    none, with_r = optimum(loops0, edges0), optimum(loops1, edges1)
    s, li = np.unravel_index(np.argmax(none - with_r), none.shape)
    before = loops0[s, li]
    after = assign(loops1[s, li], edges1[s, li])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    x = np.arange(EP)
    a1.bar(x - 0.2, before, 0.38, color=DDR, label="no replicas")
    a1.bar(x + 0.2, after, 0.38, color=HBM, label=f"with replicas (max {after.max()})")
    a1.set_xticks(x, [f"GPU {r}" for r in range(EP)])
    a1.set_ylabel("active cold experts in the layer")
    a1.set_title("One held-out layer and step", fontsize=12)
    a1.legend(loc="upper right")
    names = list(budgets)
    busiest = [budgets[k]["critical_cold_per_step"] for k in names]
    mean = budgets[names[0]]["mean_cold_per_rank_per_step"]
    a2.bar(range(len(names)), busiest, color=[DDR] + [HBM] * (len(names) - 1), width=0.6)
    a2.axhline(mean, color=INK2, linestyle="--", linewidth=1.2)
    a2.text(-0.45, mean - 4, f"mean GPU: {mean:.0f}", ha="left", va="top", color="white",
            fontsize=10, fontweight="bold")
    for i, v in enumerate(busiest):
        a2.text(i, v + 3, f"{v:.0f}\n(+{v - mean:.0f})", ha="center", fontsize=9.5, color=INK)
    a2.set_xticks(range(len(names)), names)
    a2.set_ylim(0, max(busiest) * 1.2)
    a2.set_ylabel("cold experts on the busiest GPU,\nsummed over layers, per step")
    a2.set_title("Replicas per GPU (held-out steps)", fontsize=12)
    fig.suptitle("Spare Grace copies let the busiest GPU hand cold work to a quieter one",
                 x=0.01, ha="left", fontweight="bold", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out / "glm-replicas.png", dpi=160)
    plt.close(fig)


def ab_rows(tags: list[str]) -> list[tuple]:
    rows = []
    for tag in tags:
        for f in sorted(HERE.glob(f"ab-{tag}-*.json")):
            arm = f.stem.split("-", 2)[2]
            for ctx, v in json.loads(f.read_text()).items():
                if not ctx.startswith("ttft"):
                    rows += [(tag, arm, ctx, r["step_ms"], r["tokens_per_step"])
                             for r in v["runs"]]
    return rows


def fig_ab(out: Path) -> None:
    groups = {
        "no replicas (Marlin)": ("rA", "rB", {"off"}),
        "exact replicas (Marlin)": ("rA", "rB", {"exact"}),
        "replicas + one-kernel": ("rA", "rB", {"exactok"}),
        "+ DCP4": ("dA", None, {"dcp4a", "dcp4b"}),
    }
    colors = [DDR, "#e8a33d", HBM, "#1d4f8f"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for (name, (t1, t2, arms)), color in zip(groups.items(), colors):
        rows = [r for r in ab_rows([t for t in (t1, t2) if t]) if r[1] in arms]
        x = np.array([r[4] for r in rows])
        y = np.array([r[3] for r in rows])
        ax.scatter(x, y, s=22, color=color, alpha=0.8, label=f"{name} ({len(rows)} runs)")
        k, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 10)
        ax.plot(xx, k * xx + b, color=color, linewidth=1.4)
    ax.set_xlabel("accepted tokens per step (greedy, run to run)")
    ax.set_ylabel("decode step, ms")
    ax.set_title("Every A/B run, against how many draft tokens it accepted", pad=26)
    ax.text(0, 1.02, "greedy output varies run to run and step time rises ~1.8 ms per "
            "accepted token, so arms are compared at matched acceptance",
            transform=ax.transAxes, fontsize=9.5, color=INK2)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "glm-ab-runs.png", dpi=160)
    plt.close(fig)


# (label, ms change against its own control, +- , control arm description)
LADDER = [
    ("MTP7 as served\n(draft decodes\neager)", None, None),
    ("capture the\ndraft-decode\ngraph [1, 8]", -0.65, 0.28),
    ("Grace replicas\n(exact, Marlin)", -4.09, 0.25),
    ("INT4 one-kernel,\nbalanced by\ntime", -1.42, 0.25),
    ("DCP4", -1.12, 0.25),
]
# The no-replica arm fitted at 6 accepted tokens (fit_ab.py off rA rB: 35.06 +
# 1.81 x 6 = 45.9 ms, short context), plus the capture fix it already includes.
# That arm also ran reserve 7 (+143 hot experts), which no A/B isolated.
LADDER_START = 45.92 + 0.65


def fig_ladder(out: Path) -> None:
    values, labels = [LADDER_START], [LADDER[0][0]]
    for name, delta, _ in LADDER[1:]:
        values.append(values[-1] + delta)
        labels.append(name)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(values))
    ax.bar(x, values, width=0.62, color=[MUTED] + [HBM] * (len(values) - 1),
           edgecolor=SURFACE, linewidth=2)
    for xi, v, (_, delta, err) in zip(x, values, LADDER):
        ax.text(xi, v + 0.6, f"{v:.1f} ms", ha="center", color=INK)
        if delta is not None:
            ax.text(xi, v / 2, f"{delta:+.2f}\n+-{err:.2f}", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0, 55)
    ax.set_ylabel("decode step, ms (8-token verify, at 6 accepted tokens)")
    ax.set_title("GLM-5.3 decode on one 4x GH200 node: ~129 -> ~153 tok/s on the greedy "
                 "bench\n(inside the bars: each change against its own same-node control)",
                 pad=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "glm-ladder.png", dpi=160)
    plt.close(fig)


BUCKETS = [  # (label, analyze.py categories, kind)
    ("MoE experts", ("MoE hot Marlin (HBM)", "MoE cold Marlin (Grace)", "MoE one-kernel w13",
                     "MoE one-kernel w2", "MoE one-kernel route/act/finalize",
                     "MoE routing/align/sum/act"), "moe"),
    ("all-reduce (waiting + transfer)", ("TP all-reduce",), "wait"),
    ("DCP all-gather / reduce-scatter", ("MoE all-gather (SP)", "MoE reduce-scatter (SP)"), "comm"),
    ("dense GEMMs", ("dense GEMM",), "other"),
    ("norms, rope, glue", ("norm/rope/elementwise",), "other"),
    ("attention + indexer", ("attention", "DSA indexer"), "other"),
]


def fig_breakdown(out: Path) -> None:
    runs = [("this morning: DCP1, no replicas,\nMarlin, eager draft decodes",
             "breakdown-marlin-2096362.json"),
            ("now: DCP4, replicas, INT4 one-kernel,\ncaptured draft decodes",
             "breakdown-dcp4-2097699.json")]
    palette = {"MoE experts": HBM, "all-reduce (waiting + transfer)": DDR,
               "DCP all-gather / reduce-scatter": "#e8a33d", "dense GEMMs": "#5b8a72",
               "norms, rope, glue": "#9fb8a9", "attention + indexer": "#8e6fb3",
               "drafting": "#c9c7c0", "idle / other": "#e6e5e0"}
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for yi, (label, fname) in enumerate(runs):
        data = next(iter(json.loads((HERE / fname).read_text()).values()))
        ranks = data["ranks"].values()
        mean = lambda f: float(np.mean([f(r) for r in ranks]))  # noqa: E731
        left = 0.0
        parts = [(name, sum(mean(lambda r, c=c: r["target_categories_ms"].get(c, 0))
                            for c in cats), kind) for name, cats, kind in BUCKETS]
        parts.append(("drafting", mean(lambda r: r["phase_busy_ms"].get("draft", 0)), "other"))
        period = mean(lambda r: r["period_ms"]["mean"])
        busy = sum(p[1] for p in parts)
        parts.append(("idle / other", period - busy, "other"))
        for name, ms, _ in parts:
            ax.barh(yi, ms, left=left, color=palette[name], edgecolor=SURFACE, linewidth=1.5,
                    label=name if yi == 0 else None)
            if ms > 1.2:
                ax.text(left + ms / 2, yi, f"{ms:.1f}", ha="center", va="center", fontsize=9,
                        color=INK)
            left += ms
        ax.text(left + 0.4, yi, f"{period:.1f} ms profiled", va="center", fontsize=10)
    ax.set_yticks(range(len(runs)), [r[0] for r in runs], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 66)
    ax.set_xlabel("ms per decode step (mean of 4 GPUs, short context, profiler on)")
    ax.set_title("Where one GLM-5.3 decode step goes, before and after", loc="left", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=4, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "glm-step-breakdown.png", dpi=160)
    plt.close(fig)


def fig_gemm(out: Path) -> None:
    data = json.loads((HERE / "bench-gemm-2097277.txt").read_text().splitlines()[-1])
    rows = [r for r in data["rows"] if "router" not in r["name"] and r["per_step"] > 3]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.barh(yi, r["step_ms"], color=MUTED, height=0.6)
        ax.barh(yi, r["floor_step_ms"], color=HBM, height=0.6)
        ax.text(r["step_ms"] + 0.02, yi, f"{r['floor_us'] / r['us']:.0%} of floor",
                va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [f"{r['name']}  {r['k']}x{r['n']}" for r in rows], fontsize=9)
    ax.set_xlabel("ms per decode step (M = 8, plain F.linear, CUDA graph, one GPU)")
    fig.suptitle("Verify-step projections against their HBM read floor (blue)", x=0.01,
                 ha="left", fontweight="bold", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out / "glm-gemm.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    want = lambda name: not args.only or name in args.only  # noqa: E731
    profile = json.loads(args.profile.read_text())
    layers = profile["routed_layers"]
    owners = np.asarray(profile["owners"])
    n = owners.shape[1]
    if want("skew") or want("coverage"):
        train, held = (routes(args.trace_dir, layers, n, s) for s in ("train", "heldout"))
        if want("skew"):
            fig_skew(train, args.out)
        if want("coverage"):
            fig_coverage(train, held, args.out)
    if want("residency") or want("replicas"):
        train_act = active_mask(load_steps(args.trace_dir, "train", layers), n)
        held_act = active_mask(load_steps(args.trace_dir, "heldout", layers), n)
        if want("residency"):
            fig_residency(train_act, held_act, owners, args.out)
        if want("replicas"):
            hot = np.zeros(owners.shape, dtype=bool)
            for li, ids in enumerate(profile["hot_experts"]):
                hot[li, ids] = True
            secondary = np.asarray(profile["secondary_ranks"])
            r2000 = json.loads((args.profile.parent / "glm53-w4a16-2496-r2000.json").read_text())
            budgets = {"none": evaluate(held_act, owners, hot, np.full(owners.shape, -1)),
                       "985 (served)": evaluate(held_act, owners, hot, secondary),
                       "2000": evaluate(held_act, owners, hot,
                                        np.asarray(r2000["secondary_ranks"]))}
            fig_replicas(held_act, owners, hot, secondary, budgets, args.out)
    for name, fn in (("ab", fig_ab), ("ladder", fig_ladder), ("breakdown", fig_breakdown),
                     ("gemm", fig_gemm)):
        if want(name):
            fn(args.out)
    print(sorted(p.name for p in args.out.glob("*.png")))


if __name__ == "__main__":
    main()
