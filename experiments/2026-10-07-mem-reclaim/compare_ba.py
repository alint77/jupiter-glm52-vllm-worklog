"""Before (HEAD prod) vs after (workspaces + shared NCCL comm + embedding on
Grace, reserve 1.7) from arm.sh outputs.

Per arm: hot experts, tightest-rank startup free, GSM8K, acceptance, TTFT,
stress survival and peak. Decode: step_ms ~ tokens/step + ctx + node + after
over agentic requests (>= 20 steps), and the same over the long-context rows.

    compare_ba.py [root]
"""
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/e/fscratch/profound/naeimitabiei1/mem-reclaim")
logs = Path(os.environ.get("LOGS", Path(__file__).resolve().parent))
# arm labels: tag prefixes "<a>-<job>..." / "<b>-<job>..."; the fit is b - a
A, Bk = os.environ.get("KINDS", "before,after").split(",")
arms = []
for d in sorted(root.glob(f"[{A[0]}{Bk[0]}]*-22*")):
    kind, job = d.name.split("-")[0], d.name.split("-")[1]
    log = logs / f"run-{d.name}.log"
    if kind not in (A, Bk) or not log.exists() or "=== done" not in log.read_text():
        continue
    out = (d / "server.out").read_text(errors="ignore")
    tp0 = [l for l in out.splitlines() if "Worker_TP0" in l and "residency:" in l]
    hot = int(re.search(r"residency: (\d+) hot", tp0[-1]).group(1)) if tp0 else None
    free = [float(x) for x in re.findall(r"observed HBM reserve: ([0-9.]+) GiB", out)]
    text = log.read_text()
    ttft = dict(re.findall(r"TTFT (.+?): median ([0-9.]+) s", text))
    peaks = [int(x) for x in re.findall(r"peak (\d+) MiB", text)]
    gsm = json.loads((d / "gsm8k.json").read_text())["accuracy"] if (d / "gsm8k.json").exists() else None
    rows = [json.loads(l) for l in (d / "rows.jsonl").read_text().splitlines() if l.strip()]
    long = [json.loads(l) for l in (d / "long.jsonl").read_text().splitlines() if l.strip()]
    arms.append(dict(name=d.name, kind=kind, job=job, hot=hot, free=min(free) if free else None,
                     ttft={k: float(v) for k, v in ttft.items()}, peak=max(peaks) if peaks else None,
                     alive="server alive" in text, failed=" FAILED" in text, gsm=gsm, rows=rows,
                     long=long))

print(f"{'arm':22s} {'hot':>5s} {'free':>5s} {'gsm8k':>6s} {'acc':>6s} {'tok/st':>6s} {'ms/st':>6s} "
      f"{'TTFT 20K/14K':>12s} {'8K/150K':>8s} {'60K':>6s} {'388K':>5s} {'peak MiB':>8s}")
for a in arms:
    r = a["rows"]
    acc = sum(x["accepted"] for x in r) / max(1, sum(x["draft_tokens"] for x in r))
    tps = 1 + sum(x["accepted"] for x in r) / max(1, sum(x["steps"] for x in r))
    ms = 1000 * sum(x["decode_s"] for x in r) / max(1, sum(x["steps"] for x in r))
    t = a["ttft"]
    print(f"{a['name']:22s} {a['hot'] or 0:5d} {a['free'] or 0:5.2f} {a['gsm'] or 0:6.3f} {acc:6.3f} "
          f"{tps:6.2f} {ms:6.2f} {t.get('20K new on 14K cached', 0):12.3f} "
          f"{t.get('8K new on 150K cached', 0):8.3f} {t.get('60K uncached', 0):6.2f} "
          f"{'OK' if a['alive'] and not a['failed'] else 'FAIL':>5s} {a['peak'] or 0:8d}")


def fit(rows_by_arm, label):
    if len({a["kind"] for a, _ in rows_by_arm}) < 2:
        print(f"{label}: needs both arms")
        return
    X, y, w = [], [], []
    jobs = sorted({a["job"] for a, _ in rows_by_arm})
    for a, rows in rows_by_arm:
        for r in rows:
            ctx = r.get("prompt_tokens", r.get("ctx", 0))
            x = [1.0, r["tokens_per_step"], ctx / 1e4, 1.0 if a["kind"] == Bk else 0.0]
            x += [1.0 if a["job"] == j else 0.0 for j in jobs[1:]]
            X.append(x)
            y.append(r["step_ms"])
            w.append(r["steps"])
    X, y, w = np.array(X), np.array(y), np.sqrt(np.array(w))
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    res = (y - X @ beta) * w
    dof = len(y) - X.shape[1]
    cov = np.linalg.inv((X * w[:, None]).T @ (X * w[:, None])) * (res @ res) / dof
    print(f"{label}: {Bk} - {A} = {beta[3]:+.3f} +- {np.sqrt(cov[3, 3]):.3f} ms/step "
          f"({len(y)} requests, {len(jobs)} nodes; per tok/step {beta[1]:+.2f} ms, per 10K ctx {beta[2]:+.3f} ms)")


fit([(a, [r for r in a["rows"] if r["steps"] >= 20 and "step_ms" in r]) for a in arms], "agentic decode")
fit([(a, a["long"]) for a in arms], "long-context decode (50K/130K)")
for kind in (A, Bk):
    g = [a["gsm"] for a in arms if a["kind"] == kind and a["gsm"] is not None]
    if g:
        print(f"GSM8K {kind}: mean {st.mean(g):.3f} over {len(g)} x 200")
