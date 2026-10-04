"""Exact verify-step time distribution from a VLLM_STEP_TRACE_FILE log
(lines: wall time, request id, drafts, accepted). A step's time is the gap to
the previous step of the same request; gaps over 1 s (idle between turns, or
a prefill in between) are dropped.
Usage: plot_steps.py <steps.csv> <out.png> [manifest.jsonl]"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2026-09-26-mimo-routing-profile"))
from plot_routing import HBM, INK, INK2  # noqa: E402

last, gaps, accepted = {}, [], []
for line in open(sys.argv[1]):
    if line.startswith("#") or not line.strip():
        continue
    t, rid, drafts, acc = line.strip().split(",")
    t = float(t)
    if rid in last and t - last[rid] < 1.0:
        gaps.append(1e3 * (t - last[rid]))
        accepted.append(int(acc) + 1)
    last[rid] = t
ms = np.array(gaps)
q = {k: np.percentile(ms, p) for k, p in (("p1", 1), ("p50", 50), ("p99", 99))}
print(f"{len(ms)} steps: min {ms.min():.1f} p1 {q['p1']:.1f} median {q['p50']:.1f} "
      f"mean {ms.mean():.1f} p99 {q['p99']:.1f} max {ms.max():.1f} ms; "
      f"tokens/step {np.mean(accepted):.2f}")

if len(sys.argv) > 3:
    recs = [json.loads(x) for x in open(sys.argv[3]) if x.strip()]
    tps = sorted(r["output_tokens"] / r["decode_time"] for r in recs
                 if r.get("decode_time") and r["routed_positions"] >= 80)
    if tps:
        print(f"per-request decode tok/s ({len(tps)} requests >= 10 steps): min {tps[0]:.0f} "
              f"median {tps[len(tps) // 2]:.0f} max {tps[-1]:.0f}")

fig, ax = plt.subplots(figsize=(8.5, 4.4))
hi = max(60.0, float(np.ceil(np.percentile(ms, 99.9))) + 2)
ax.hist(ms, bins=np.arange(0, hi + 0.5, 0.5), color=HBM)
for lab, v in (("min", ms.min()), ("median", q["p50"]), ("p99", q["p99"]), ("max", ms.max())):
    ax.axvline(v, color=INK2, linestyle="--", linewidth=1)
    ax.text(v, ax.get_ylim()[1] * 0.97, f" {lab}\n {v:.1f}", color=INK, fontsize=9, va="top")
ax.set_xlim(0, hi)
ax.set_xlabel("verify-step time, ms (0.5 ms bins)")
ax.set_ylabel("steps")
ax.set_title(f"GLM-5.3 decode verify-step time ({len(ms):,} steps, "
             f"{np.mean(accepted):.2f} tokens/step)", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=160)
