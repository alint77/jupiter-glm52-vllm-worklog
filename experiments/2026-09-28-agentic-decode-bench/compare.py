#!/usr/bin/env python3
"""GLM against MiMo (or any arms) from agentic_bench.py rows.

Per arm: decode-step-weighted means (what a user sees), and a least-squares
fit step_ms ~ a + b * tokens_per_step + c * context/1e4 over requests with at
least --min-steps steps, evaluated at a common point so arms compare at matched
acceptance and context.

    compare.py rows-glm-tasks2.jsonl rows-mimo-tasks2.jsonl [--at-tps 3.5 --at-ctx 8000]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path, min_steps: int, cap: int = 8192) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = [r for r in rows if "step_ms" in r and r["steps"] >= min_steps
           and not r.get("profiled")]
    capped = [r for r in out if (r.get("completion_tokens") or r.get("output") or 0) >= cap]
    if capped:
        print(f"{path.stem}: dropping {len(capped)} request(s) that ran to the "
              f"{cap}-token cap (looping text)")
    return [r for r in out if r not in capped]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=Path, nargs="+")
    ap.add_argument("--min-steps", type=int, default=20)
    ap.add_argument("--at-tps", type=float, default=3.5)
    ap.add_argument("--at-ctx", type=float, default=8000)
    args = ap.parse_args()
    for path in args.rows:
        rows = load(path, args.min_steps)
        steps = np.array([r["steps"] for r in rows])
        ms = np.array([r["step_ms"] for r in rows])
        tps = np.array([r["tokens_per_step"] for r in rows])
        ctx = np.array([r["context"] if "context" in r else r["prompt_tokens"] for r in rows],
                       dtype=float)
        w = steps / steps.sum()
        X = np.stack([np.ones_like(ms), tps, ctx / 1e4], 1)
        coef, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], ms * np.sqrt(w), rcond=None)
        res = ms - X @ coef
        at = coef @ [1, args.at_tps, args.at_ctx / 1e4]
        tok_s = (w * tps).sum() / (w * ms).sum() * 1000
        print(f"{path.stem}: {len(rows)} requests, {steps.sum():.0f} steps, "
              f"context {np.median(ctx):.0f} median ({ctx.max():.0f} max)")
        print(f"  step-weighted: {(w * ms).sum():.2f} ms/step at {(w * tps).sum():.2f} "
              f"tokens/step -> {tok_s:.0f} tok/s decode")
        print(f"  fit: {coef[0]:.2f} + {coef[1]:.2f}/token + {coef[2]:.2f}/10K ctx "
              f"(resid sd {res.std():.2f}); at {args.at_tps} tok, {args.at_ctx:.0f} ctx: "
              f"{at:.2f} ms")


if __name__ == "__main__":
    main()
