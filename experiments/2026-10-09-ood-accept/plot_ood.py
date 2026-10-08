#!/usr/bin/env python3
"""glm-drafter-ood.png for the write-up: acceptance per prompt (answer mode,
thinking off) for MTP3 and DFlash2 k=3, and acceptance per draft position.

    plot_ood.py <out dir>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26-mimo-routing-profile"))
from plot_routing import DDR, GRID, HBM, INK, INK2, MUTED, SURFACE  # noqa: E402

ROOT = Path("/e/fscratch/profound/naeimitabiei1/ood-accept")
MODE = "content"


def rows(arm):
    f = ROOT / arm / "acc.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip() and json.loads(l)["mode"] == MODE]


def pooled(rs):
    d = sum(r["drafts"] for r in rs)
    return 1 + sum(r["accepted"] for r in rs) / d


def positions(rs):
    d = sum(r["drafts"] for r in rs)
    out = {}
    for r in rs:
        for k, v in r["pos_rate"].items():
            out[int(k)] = out.get(int(k), 0) + v * r["drafts"]
    return [out[k] / d for k in sorted(out)]


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    out = Path(sys.argv[1])
    mtp, df = rows("mtp3"), rows("dflash2-k3")
    prompts = json.loads((HERE / "prompts.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [2.6, 1]})
    x0, ticks, labels, seps = 0, [], [], []
    for dom, name in (("code", "code"), ("math", "maths"), ("prose", "prose"), ("other", "other")):
        ids = [p["id"] for p in prompts if p["domain"] == dom]
        for pid in ids:
            m = pooled([r for r in mtp if r["id"] == pid])
            d = pooled([r for r in df if r["id"] == pid])
            a1.bar(x0 - 0.2, m, 0.38, color=MUTED, edgecolor=SURFACE)
            a1.bar(x0 + 0.2, d, 0.38, color=HBM, edgecolor=SURFACE)
            ticks.append(x0)
            labels.append(pid)
            x0 += 1
        dm = pooled([r for r in mtp if r["domain"] == dom])
        dd = pooled([r for r in df if r["domain"] == dom])
        mid = x0 - (len(ids) + 1) / 2
        a1.text(mid, 3.45, f"{name}\n{dd / dm - 1:+.0%}", ha="center", fontsize=9.5, color=INK,
                fontweight="bold")
        seps.append(x0 - 0.5)
        x0 += 0.6
    for s in seps[:-1]:
        a1.axvline(s + 0.3, color=GRID, linewidth=1)
    a1.set_xticks(ticks)
    a1.set_xticklabels(labels, rotation=55, ha="right", fontsize=8.5, color=INK)
    a1.set_ylim(1, 3.75)
    a1.set_ylabel("tokens per step (accepted + 1, of 4)", color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MUTED, HBM)]
    a1.legend(handles, ["MTP3", "DFlash2 k=3"], loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=2, frameon=False, fontsize=9.5)
    a1.set_title("Per prompt (3 seeds); DFlash2 vs MTP3 per domain", loc="left",
                 fontsize=10.5, color=INK)
    style(a1)
    pm, pd = positions(mtp), positions(df)
    xs = range(1, len(pm) + 1)
    a2.plot(xs, pm, "o-", color=MUTED, label="MTP3: accepted at i / drafts")
    a2.plot(xs, pd, "o-", color=HBM, label="DFlash2 k=3")
    cm = [pm[0]] + [pm[i] / pm[i - 1] for i in range(1, len(pm))]
    cd = [pd[0]] + [pd[i] / pd[i - 1] for i in range(1, len(pd))]
    a2.plot(xs, cm, "o--", color=MUTED, alpha=0.7, label="MTP3: given i-1 accepted")
    a2.plot(xs, cd, "o--", color=DDR, alpha=0.8, label="DFlash2: given i-1 accepted")
    for i, (u, v) in enumerate(zip(pm, pd)):
        a2.text(i + 1.08, u, f"{u:.2f}", fontsize=8.5, color=INK2, va="center")
        a2.text(i + 1.08, v - 0.03, f"{v:.2f}", fontsize=8.5, color=HBM, va="center")
    a2.set_xticks(list(xs))
    a2.set_xlabel("draft position", color=INK2)
    a2.set_ylabel("acceptance rate", color=INK2)
    a2.set_ylim(0.2, 0.8)
    a2.legend(loc="lower left", frameon=False, fontsize=8)
    a2.set_title("Per draft position, all prompts", loc="left", fontsize=10.5, color=INK)
    style(a2, "both")
    fig.suptitle("DFlash2 falls behind MTP3 the further the text is from code; the gap is its "
                 "first draft token", x=0.01, ha="left", fontsize=12.5, fontweight="bold",
                 color=INK)
    fig.text(0.01, 0.885, "16 prompts, answer mode (thinking off), temperature 1.0 / top_p 0.95, "
             "512 tokens, one request at a time; same 3-token draft length", fontsize=9.5,
             color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out / "glm-drafter-ood.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
