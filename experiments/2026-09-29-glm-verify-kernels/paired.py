#!/usr/bin/env python3
"""Paired A/B of agentic_bench.py rows: arm B against arm A on the same requests.

Rows are matched by request key; requests with fewer than --min-steps decode
steps, profiled ones and ones that hit the token cap are dropped. Reports the
step-weighted means of both arms on the common set and a joint least-squares
fit  step_ms ~ a + d*[arm B] + b*tokens_per_step + c*ctx/1e4  whose d is the
matched-acceptance, matched-context step-time difference, with a bootstrap
interval over requests.

    paired.py --a rows-x.jsonl [rows-x2.jsonl] --b rows-y.jsonl [rows-y2.jsonl]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(paths, min_steps, cap=8192):
    rows = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if ("step_ms" not in r or r["steps"] < min_steps or r.get("profiled")
                    or (r.get("completion_tokens") or 0) >= cap):
                continue
            rows.setdefault(r["key"], []).append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--min-steps", type=int, default=20)
    args = ap.parse_args()
    A, B = load(args.a, args.min_steps), load(args.b, args.min_steps)
    keys = sorted(set(A) & set(B))
    rows = [(0, r) for k in keys for r in A[k]] + [(1, r) for k in keys for r in B[k]]
    arm = np.array([a for a, _ in rows], dtype=float)
    steps = np.array([r["steps"] for _, r in rows], dtype=float)
    ms = np.array([r["step_ms"] for _, r in rows])
    tps = np.array([r["tokens_per_step"] for _, r in rows])
    ctx = np.array([r.get("context", r["prompt_tokens"]) for _, r in rows], dtype=float)
    for label, m in (("A", arm == 0), ("B", arm == 1)):
        w = steps[m] / steps[m].sum()
        print(f"{label}: {m.sum()} requests, {steps[m].sum():.0f} steps: "
              f"{(w * ms[m]).sum():.3f} ms/step at {(w * tps[m]).sum():.3f} tokens/step "
              f"-> {(w * tps[m]).sum() / (w * ms[m]).sum() * 1000:.1f} tok/s")
    X = np.stack([np.ones_like(ms), arm, tps, ctx / 1e4], 1)

    def fit(idx):
        w = np.sqrt(steps[idx])
        coef, *_ = np.linalg.lstsq(X[idx] * w[:, None], ms[idx] * w, rcond=None)
        return coef

    coef = fit(np.arange(len(ms)))
    rng = np.random.default_rng(0)
    groups = {}
    for i, (_, r) in enumerate(rows):
        groups.setdefault(r["key"], []).append(i)
    glist = list(groups.values())
    boots = []
    for _ in range(1000):
        pick = rng.integers(0, len(glist), len(glist))
        boots.append(fit(np.concatenate([glist[p] for p in pick]))[1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{len(keys)} common requests; fit: B - A = {coef[1]:+.3f} ms/step "
          f"(95% CI {lo:+.3f} .. {hi:+.3f}); {coef[2]:+.3f}/token, "
          f"{coef[3]:+.3f}/10K ctx, intercept {coef[0]:.2f}")


if __name__ == "__main__":
    main()
