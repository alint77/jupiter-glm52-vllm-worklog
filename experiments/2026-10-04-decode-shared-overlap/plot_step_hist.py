"""Decode verify-step time distribution of a running server, from its
Prometheus histogram (vllm:inter_token_latency_seconds: one sample per verify
step under speculative decoding). The buckets are coarse, so bars span them.
Usage: plot_step_hist.py <metrics.txt> <out.png>"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2026-09-26-mimo-routing-profile"))
from plot_routing import HBM, INK, INK2, MUTED  # noqa: E402

text = Path(sys.argv[1]).read_text()
buckets = []
for line in text.splitlines():
    if line.startswith("vllm:inter_token_latency_seconds_bucket"):
        le = re.search(r'le="([^"]+)"', line).group(1)
        buckets.append((float("inf") if le == "+Inf" else float(le), float(line.split()[-1])))
count = float(re.search(r"^vllm:inter_token_latency_seconds_count\S* ([0-9.e+]+)$", text, re.M).group(1))
total = float(re.search(r"^vllm:inter_token_latency_seconds_sum\S* ([0-9.e+]+)$", text, re.M).group(1))
mean_ms = 1e3 * total / count

edges, counts, prev_le, prev_c = [], [], 0.0, 0.0
for le, c in buckets:
    if le == float("inf"):
        break
    edges.append((prev_le * 1e3, le * 1e3))
    counts.append(c - prev_c)
    prev_le, prev_c = le, c

fig, ax = plt.subplots(figsize=(8.5, 4.4))
shown = [(lo, hi, c) for (lo, hi), c in zip(edges, counts) if hi <= 100]
for lo, hi, c in shown:
    share = 100 * c / count
    ax.bar(lo, share, width=hi - lo, align="edge", color=HBM if c else MUTED,
           edgecolor="white", linewidth=1.5)
    if c:
        ax.text((lo + hi) / 2, share + 1.5, f"{share:.0f}%\n({c:,.0f} steps)",
                ha="center", fontsize=9.5, color=INK)
ax.axvline(mean_ms, color=INK2, linestyle="--", linewidth=1.3)
ax.text(mean_ms + 1, 92, f"mean {mean_ms:.1f} ms", color=INK2, fontsize=9.5)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xticks([0, 10, 25, 50, 75, 100])
ax.set_xlabel("verify-step time, ms (server histogram buckets: 0-10, 10-25, 25-50, 50-75, 75-100)")
ax.set_ylabel("share of verify steps, %")
ax.set_title(f"GLM-5.3 decode step time, live Claude Code session ({count:,.0f} steps)\n"
             "DFlash2 8-token verify, DCP4, production config with routing capture",
             loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=160)
