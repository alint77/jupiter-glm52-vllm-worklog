#!/usr/bin/env python3
"""Figures for the M <= 32 work (write-up glm/README.md section 10).

    plot_m32.py <out dir>

glm-m32-moe.png          MoE layer time per step size: decode kernel vs the
                         path it replaced (grid-*.jsonl, m8-*.jsonl)
glm-m32-ab.png           served step time before / after at 8 / 16 / 32
                         tokens (c=4 k=7 A/B, compare_conc.py's pairing)
glm-concurrency.png      total and per-request decode tok/s vs requests in
                         flight, every sweep config (m32-sweep/*/sweep.jsonl)
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26-mimo-routing-profile"))
from plot_routing import DDR, GRID, HBM, INK, INK2, MUTED, SURFACE  # noqa: E402

SCRATCH = Path("/e/fscratch/profound/naeimitabiei1")
PROD_C1 = 165  # tok/s, c=1 DFlash2 k=7 prod (2026-10-08-flashmla-split A/B)


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def finish(fig, title, subtitle, path, top=0.82):
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, top + 0.035, subtitle, fontsize=9.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def weighted(path, kern):
    by = collections.defaultdict(list)
    w = {}
    for line in open(path):
        r = json.loads(line)
        if r["kernel"] == kern:
            by[(r["hot"], r["cold"])].append(r["us"])
            w[(r["hot"], r["cold"])] = r["weight"]
    tot = sum(w.values())
    return sum(w[k] * min(v) for k, v in by.items()) / tot


def moe(out):
    g = sorted(HERE.glob("grid-22*-m16.jsonl"))[0].name.split("-m16")[0]
    rows = [("8", weighted(sorted(HERE.glob("m8-*.jsonl"))[0], "decode@HEAD"),
             weighted(sorted(HERE.glob("m8-*.jsonl"))[0], "decode"), "decode kernel (8-token build)"),
            ("16", weighted(HERE / f"{g}-m16.jsonl", "prefill"),
             weighted(HERE / f"{g}-m16.jsonl", "decode"), "wgmma prefill kernel"),
            ("32", weighted(HERE / f"{g}-m32.jsonl", "prefill"),
             weighted(HERE / f"{g}-m32.jsonl", "decode"), "wgmma prefill kernel")]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    for i, (m, before, after, what) in enumerate(rows):
        ax.bar(i - 0.2, before, 0.38, color=MUTED, edgecolor=SURFACE)
        ax.bar(i + 0.2, after, 0.38, color=HBM, edgecolor=SURFACE)
        ax.text(i - 0.2, before + 4, f"{before:.0f}", ha="center", fontsize=9.5, color=INK)
        ax.text(i + 0.2, after + 4, f"{after:.0f}", ha="center", fontsize=9.5, color=INK)
        if before - after > 2:
            ax.text(i + 0.2, after / 2, f"{(after / before - 1) * 100:.0f}%", ha="center",
                    color="white", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r[0]} tokens per step\nbefore: {r[3]}" for r in rows], fontsize=9.5,
                       color=INK)
    ax.set_ylabel("us per MoE layer (weighted over\nthe cells covering 95% of calls)", color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MUTED, HBM)]
    ax.legend(handles, ["before", "tiered decode kernel, up to 32 tokens"], loc="upper left",
              frameon=False, fontsize=9.5)
    ax.set_ylim(0, max(r[1] for r in rows) * 1.18)
    style(ax)
    finish(fig, "MoE layer at 16 and 32 tokens: the decode kernel instead of the prefill kernel",
           "per GPU, INT4, live-routing (hot, cold) mix at each size; 8 tokens: unchanged",
           out / "glm-m32-moe.png")


def ab(out):
    cell = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in sorted((SCRATCH / "m32").glob("m*-*")):
        kind, job = d.name.split("-")[:2]
        log = HERE / f"run-{d.name}.log"
        if not (d / "probe.jsonl").exists() or "=== done" not in log.read_text():
            continue
        for line in (d / "probe.jsonl").read_text().splitlines():
            r = json.loads(line)
            cell[8 * r["n"]][(kind, job)].append(1000 * r["acc_len"] / r["decode_tps"])
    sizes = sorted(cell)
    before, after = [], []
    for m in sizes:
        b = [st.mean(v) for (k, _), v in cell[m].items() if k == "m8"]
        a = [st.mean(v) for (k, _), v in cell[m].items() if k == "m32"]
        before.append(st.mean(b))
        after.append(st.mean(a))
    fig, ax = plt.subplots(figsize=(8.5, 4.0), facecolor=SURFACE)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.bar(i - 0.2, b, 0.38, color=MUTED, edgecolor=SURFACE)
        ax.bar(i + 0.2, a, 0.38, color=HBM, edgecolor=SURFACE)
        ax.text(i - 0.2, b + 0.6, f"{b:.1f}", ha="center", fontsize=9.5, color=INK)
        ax.text(i + 0.2, a + 0.6, f"{a:.1f}", ha="center", fontsize=9.5, color=INK)
        if b - a > 0.3:
            ax.text(i + 0.2, a / 2, f"{a - b:+.1f} ms", ha="center", color="white", fontsize=9.5,
                    fontweight="bold", rotation=90)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{m // 8} in flight\n({m} tokens)" for m in sizes], fontsize=10,
                       color=INK)
    ax.set_ylabel("ms per decode step", color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MUTED, HBM)]
    ax.legend(handles, ["before (8-token decode paths)", "after (up to 32)"], loc="upper left",
              frameon=False, fontsize=9.5)
    ax.set_ylim(0, max(before) * 1.15)
    style(ax)
    finish(fig, "Served: the 16- and 32-token steps get ~10% faster",
           "c=4, DFlash2 k=7, 5K-50K contexts; same-node pairs, step = accepted tokens / "
           "per-request tok/s", out / "glm-m32-ab.png")


CONFIGS = [  # tag, label, color, linestyle
    ("dflash23-c2", "c=2, 400K", "#9cc3ef", "-"),
    ("dflash23-c4", "c=4, 400K", HBM, "-"),
    ("dflash23-c8-200k", "c=8, 200K", "#123f7a", "-"),
    ("dflash23-c8-400k-r500", "c=8, 400K, 500 replicas", DDR, "-"),
    ("dflash23-c8-400k-norep", "c=8, 400K, no replicas", "#f2a37f", "-"),
]


def sweep_rows(tag):
    f = SCRATCH / "m32-sweep" / tag / "sweep.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    out = {}
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        out[n] = (st.mean(r["agg_tps"] for r in rs), st.mean(r["decode_tps"] for r in rs))
    return out


def concurrency(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4), facecolor=SURFACE)
    for spec, ls, mk in (("dflash2", "-", "o"), ("mtp", "--", "s")):
        for tag, label, color, _ in CONFIGS:
            t = tag.replace("dflash23", f"{spec}3")
            d = sweep_rows(t)
            if not d:
                continue
            ns = sorted(d)
            lab = f"{'DFlash2 k=3' if spec == 'dflash2' else 'MTP3'}, {label}"
            a1.plot(ns, [d[n][0] for n in ns], ls, marker=mk, color=color, label=lab, markersize=5)
            a2.plot(ns, [d[n][1] for n in ns], ls, marker=mk, color=color, markersize=5)
    for ax in (a1, a2):
        ax.axhline(PROD_C1, color=INK2, linestyle=":", linewidth=1.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_xlabel("requests in flight", color=INK2)
        style(ax, "both")
    a2.text(1.05, PROD_C1 - 9, "prod, c=1 (DFlash2 k=7)", fontsize=8.5, color=INK2)
    a1.set_ylabel("total decode tok/s", color=INK2)
    a2.set_ylabel("decode tok/s per request", color=INK2)
    a1.set_title("Throughput", loc="left", fontsize=10.5, color=INK)
    a2.set_title("Per request", loc="left", fontsize=10.5, color=INK)
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
               fontsize=7.8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("More users: c=8 reaches ~435 tok/s at 200K, ~370 at 400K (500 replicas)", x=0.01,
                 ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.885, "GLM-5.3 W4A16, 4x GH200, max_num_seqs = c, 4 tokens verified per "
             "request; 5K and 50K contexts averaged; solid DFlash2 k=3, dashed MTP3",
             fontsize=9.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 0.85))
    fig.savefig(out / "glm-concurrency.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for f in (moe, ab, concurrency):
        f(out)


if __name__ == "__main__":
    main()
