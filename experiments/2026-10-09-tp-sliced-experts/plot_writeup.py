#!/usr/bin/env python3
"""Write-up figures for the TP-sliced MoE at up to 16 requests (glm/README.md).

    plot_writeup.py <out dir>

glm-concurrency.png   throughput per GPU vs interactivity: the section-10 EP
                      configs plus TP-sliced MTP3 at c=8 (sweep_ab.py arms)
                      and c=16 (c16_arm.sh scale.jsonl, n = 1..16)
glm-offload.png       offloading vs every expert in HBM: tiered / all-hot
                      kernel time (roofline and measured, e78.sh), and the
                      cost per decode step (nsys traces x the bench ratio)
glm-moe-headroom.png  decode step today vs with a roofline MoE kernel
                      (max(hot / 3.6 TB/s, cold / 0.419 TB/s) per call, 75 calls)
"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-08-m32"))
from plot_m32 import (CONFIGS, DDR, GPUS, HBM, INK, INK2, MUTED, SPECS, SURFACE,  # noqa: E402
                      frontier, style, sweep_rows)
from sweep_ab import cells, collect  # noqa: E402

SL = "#1a9a62"
SL2 = "#8fd1b0"
PROF = Path("/e/fscratch/profound/naeimitabiei1/sliced-prof")
C16_RUNS = ("c16-mtp3-r4.0-2266124", "c16-mtp3-r4.0-2266125")

# sl_mix.py: mean distinct (hot, cold) experts per layer call, MTP3, n requests
MIX = {1: (21.77, 2.53), 4: (69.42, 9.37), 8: (108.37, 17.42), 16: (147.19, 29.83)}
EXPERT_B, SHARED_B = 5308416, 18874368  # int4 slice + scales; bf16 shared slice
ROOF_HBM, ROOF_C2C = 3600e9, 419e9
BENCH = {1: (4, "22,3", "25,0"), 4: (16, "69,9", "78,0"), 8: (32, "108,17", "125,0"),
         16: (64, "147,30", "177,0")}
# nsys windows (nsys_moe_calls.py): n -> (report index, step period ms from nsys_bd.py)
TRACE = {1: ("1", 17.4), 4: ("2", 27.8), 8: ("3", 37.6)}
C16_STEP_MS = 58.3  # c16 servers, n=16 at 5K


def c16_cells():
    rows = []
    for r in C16_RUNS:
        f = PROF / r / "scale.jsonl"
        if f.exists():
            rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return cells(rows)


def concurrency(out):
    _, agg, _ = collect()
    c16 = c16_cells()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0), facecolor=SURFACE, sharey=True)
    for ax, ctx in zip(axes, (5000, 50000)):
        pts = []
        for spec, name, ls, mk in SPECS:
            for tag, label, color in CONFIGS:
                d = sweep_rows(f"{spec}-{tag}", ctx)
                if not d:
                    continue
                ns = sorted(d)
                xs, ys = [d[n][0] for n in ns], [d[n][1] for n in ns]
                pts += zip(xs, ys)
                main = tag == CONFIGS[0][0]
                ax.plot(xs, ys, ls, marker=mk, color=color, markersize=5 if main else 4,
                        linewidth=1.8 if main else 1.1, alpha=1.0 if main else 0.75,
                        label=f"EP, {name}, {label}", zorder=3)
        for cs, label, color, lw in (
                ({n: agg["sl"][(ctx, n)] for c, n in agg["sl"] if c == ctx}, "TP-sliced, MTP3, c=8, 1.6M pool",
                 SL2, 2.0),
                ({n: c16[(c, n)] for c, n in c16 if c == ctx}, "TP-sliced, MTP3, c=16, 1.6M pool", SL, 2.8)):
            ns = sorted(cs)
            xs, ys = [cs[n]["user"] for n in ns], [cs[n]["gpu"] for n in ns]
            pts += zip(xs, ys)
            ax.plot(xs, ys, "--", marker="D", color=color, markersize=6, linewidth=lw, label=label,
                    zorder=4)
            if color == SL:
                for n, x, y in zip(ns, xs, ys):
                    ax.annotate(f"{n}", (x, y), xytext=(6, 4), textcoords="offset points",
                                fontsize=9.5, color=INK, fontweight="bold")
        f = frontier(pts)
        ax.plot([p[0] for p in f], [p[1] for p in f], color=MUTED, linewidth=9, alpha=0.3,
                solid_capstyle="round", zorder=1, label="Pareto frontier")
        ax.set_title(f"{ctx // 1000}K tokens of context per request", loc="left", fontsize=10.5,
                     color=INK)
        ax.set_xlabel("interactivity: decode tok/s per user", color=INK2)
        ax.set_xlim(40, 205)
        style(ax, "both")
    axes[0].set_ylabel(f"throughput: output tok/s per GPU ({GPUS} GH200)", color=INK2)
    axes[0].set_ylim(0, 190)
    axes[0].text(0.98, 0.97, "numbers: requests in flight (c=16 line)", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=8.5, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Throughput vs interactivity at 400K max context: TP-sliced at c=16 is the new "
                 "frontier", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.915, "GLM-5.3 W4A16, TP4 / DCP4; 400 output tokens, temperature 1.0; "
             "output tok/s includes TTFT on a cached prompt; solid DFlash2 k=3, dashed MTP3",
             fontsize=9.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(out / "glm-concurrency.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def bench_us(m, cell):
    h, c = map(int, cell.split(","))
    us = [json.loads(l)["us"] for l in (HERE / f"logs/ab-e78-{m}.jsonl").read_text().splitlines()
          if l.strip() and (json.loads(l)["hot"], json.loads(l)["cold"]) == (h, c)]
    return st.median(us)


def trace_us(idx):
    d = json.loads((HERE / "logs/nsys-moe-calls-sl.json").read_text())
    return st.mean(v["mean_us"] for k, v in d.items() if f"nsys.{idx}.nsys-rep" in k)


def offload_rows():
    rows = []
    for n, (h, c) in MIX.items():
        hb, cb = h * EXPERT_B + SHARED_B, c * EXPERT_B
        ideal = max(hb / ROOF_HBM, cb / ROOF_C2C) / ((hb + cb) / ROOF_HBM)
        m, mix, hot = BENCH[n]
        ratio = bench_us(m, mix) / bench_us(m, hot)
        if n in TRACE:
            t = trace_us(TRACE[n][0])
            cost, step = 75 * t * (1 - 1 / ratio) / 1e3, TRACE[n][1]
        else:
            cost, step = 75 * (bench_us(m, mix) - bench_us(m, hot)) / 1e3, C16_STEP_MS
        rows.append(dict(n=n, ideal=ideal, measured=ratio, cost_ms=cost, step_ms=step,
                         cold_share=cb / (hb + cb), traced=n in TRACE))
    return rows


def offload(out):
    rows = offload_rows()
    print(json.dumps(rows, indent=1))
    ns = [r["n"] for r in rows]
    x = range(len(ns))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.9), facecolor=SURFACE)
    a1.axhline(1.0, color=INK2, linewidth=1.0, zorder=1)
    a1.text(len(ns) - 0.6, 0.99, "same speed as all experts in HBM", ha="right", va="top",
            fontsize=8.5, color=INK2)
    a1.plot(x, [r["ideal"] for r in rows], "--o", color=MUTED, linewidth=2, markersize=7,
            label="ideal (both links at full speed)", zorder=3)
    a1.plot(x, [r["measured"] for r in rows], "-o", color=SL, linewidth=2.6, markersize=8,
            label="measured kernel", zorder=4)
    for i, r in zip(x, rows):
        a1.annotate(f"{(r['measured'] - 1) * 100:+.0f}%", (i, r["measured"]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9.5, color=SL,
                    fontweight="bold")
        a1.annotate(f"{(r['ideal'] - 1) * 100:+.0f}%", (i, r["ideal"]), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8.5,
                    color=MUTED)
    a1.set_xticks(list(x), [f"{n}\n({r['cold_share'] * 100:.0f}% cold)" for n, r in zip(ns, rows)])
    a1.set_xlabel("requests in flight (share of expert bytes read from Grace)", color=INK2)
    a1.set_ylabel("MoE kernel time, offloaded / all in HBM", color=INK2)
    a1.set_ylim(0.8, 1.5)
    a1.set_xlim(-0.45, len(ns) - 0.6)
    a1.set_title("Below the line, offloading would beat HBM alone", loc="left", fontsize=10.5,
                 color=INK)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    style(a1, "y")
    bars = a2.bar(x, [r["cost_ms"] for r in rows], color=[SL if r["traced"] else SL2 for r in rows],
                  width=0.6)
    for b, r in zip(bars, rows):
        a2.annotate(f"{max(r['cost_ms'], 0):.1f} ms\n{max(r['cost_ms'], 0) / r['step_ms'] * 100:.0f}% of "
                    f"the step", (b.get_x() + b.get_width() / 2, max(r["cost_ms"], 0)),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK)
    a2.set_xticks(list(x), [str(n) + ("" if r["traced"] else "\n(bench only)")
                            for n, r in zip(ns, rows)])
    a2.set_xlabel("requests in flight", color=INK2)
    a2.set_ylabel("extra time per decode step (ms)", color=INK2)
    a2.set_ylim(0, max(r["cost_ms"] for r in rows) * 1.35)
    a2.set_title("What offloading costs per decode step", loc="left", fontsize=10.5, color=INK)
    style(a2, "y")
    fig.suptitle("Offloading is nearly free up to 8 requests, ~10% at 16", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.885, "TP-sliced MoE kernel, MTP3; same work timed with the cold experts "
             "on Grace vs in HBM; ideal from 3.6 TB/s HBM and 0.42 TB/s C2C", fontsize=9.5,
             color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "glm-offload.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def headroom_rows():
    rows = []
    for n, (h, c) in MIX.items():
        hb, cb = h * EXPERT_B + SHARED_B, c * EXPERT_B
        m, mix, _ = BENCH[n]
        moe = 75 * bench_us(m, mix) / 1e3
        roof = 75 * max(hb / ROOF_HBM, cb / ROOF_C2C) * 1e3
        step = TRACE[n][1] if n in TRACE else C16_STEP_MS
        rows.append(dict(n=n, step=step, moe=moe, roof=roof, rest=step - moe,
                         gain=(moe - roof) / step, c2c_bound=cb / ROOF_C2C > hb / ROOF_HBM))
    return rows


def headroom(out):
    rows = headroom_rows()
    print(json.dumps(rows, indent=1))
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    y = range(len(rows))
    ax.barh(y, [r["rest"] for r in rows], color=MUTED, alpha=0.45, height=0.6,
            label="rest of the step (attention, GEMMs, all-reduce, drafter)")
    ax.barh(y, [r["roof"] for r in rows], left=[r["rest"] for r in rows], color=SL, height=0.6,
            label="MoE at the roofline (HBM 3.6 TB/s, C2C 0.42 TB/s)")
    ax.barh(y, [r["moe"] - r["roof"] for r in rows], left=[r["rest"] + r["roof"] for r in rows],
            color=SL2, height=0.6, hatch="//", edgecolor=SURFACE, label="MoE kernel's gap to the roofline")
    for i, r in zip(y, rows):
        ax.annotate(f"{r['step']:.0f} ms: {r['gain'] * 100:.0f}% above the roofline", (r["step"], i),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5,
                    color=INK, fontweight="bold")
    ax.set_yticks(list(y), [f"{r['n']} request{'s' if r['n'] > 1 else ''}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("decode step (ms)", color=INK2)
    ax.set_xlim(0, max(r["step"] for r in rows) * 1.42)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2,
              bbox_to_anchor=(0.45, -0.2))
    style(ax, "x")
    fig.suptitle("A perfect MoE kernel would make the step only 5-10% faster", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.875, "TP-sliced, MTP3, as served (cold experts on Grace); roofline = the slower "
             "of the HBM and C2C reads at full speed, 75 MoE layers", fontsize=9.5, color=INK2,
             va="bottom")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out / "glm-moe-headroom.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    offload(out)
    headroom(out)
    if all((PROF / r / "scale.jsonl").exists() for r in C16_RUNS):
        concurrency(out)


if __name__ == "__main__":
    main()
