"""Fit us = a + b * hot + c * cold per (variant, M) from bench-*.jsonl, then
replay agentic held-out steps (prod profile, 3,670 hot, replicas + deployed
balancer for EP): MoE ms per step, EP = slowest GPU per layer (whole experts
at that GPU's counts), TP-sliced = node-wide counts on the slice kernel.

    analyze.py bench-<job>.jsonl [--steps 2000]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "2026-09-26-mimo-routing-profile"))
sys.path.insert(0, str(EXP / "2026-10-09-expert-coupling"))
from coupling import PROFILE, pg, runtime_hot  # noqa: E402
from mimo_replicas import load_steps  # noqa: E402

AGENTIC = Path("/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged")


def fits(files):
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        for l in f.read_text().splitlines():
            r = json.loads(l)
            rows[(r["variant"], r["m"])][(r["hot"], r["cold"])].append(r["us"])
    out = {}
    for key, cells in sorted(rows.items()):
        X = np.array([[1, h, c] for h, c in cells], float)
        y = np.array([min(v) for v in cells.values()])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = np.abs(X @ b - y).max()
        out[key] = b
        print(f"{key[0]:12s} M={key[1]:2d}: us = {b[0]:6.1f} + {b[1]:5.2f} hot + {b[2]:5.2f} cold"
              f"   ({len(cells)} cells, max residual {res:.1f} us)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("bench", nargs="+", help="bench-<job>.jsonl (one node)")
    a = ap.parse_args()
    fit = fits([Path(f) for f in a.bench])
    _, _, _, _, cost, pair, delta, target = pg.configuration()
    prof = json.loads(PROFILE.read_text())
    layers = prof["routed_layers"]
    primary = np.asarray(prof["owners"], dtype=np.int32)
    secondary = np.asarray(prof["secondary_ranks"], dtype=np.int32)
    hot = runtime_hot(prof, 3670)
    steps = load_steps(AGENTIC, "heldout", layers)  # [n, 8, L, 8]
    rng = np.random.default_rng(0)
    for m in (8, 32):
        full = [k for k in fit if k[1] == m and k[0].startswith("full")]
        sl = [k for k in fit if k[1] == m and k[0].startswith("slice")]
        if not full or not sl:
            continue
        ep = collections.Counter()
        tp = collections.Counter()
        for _ in range(a.steps):
            idx = rng.choice(len(steps), m // 8, replace=False)
            r = np.concatenate([steps[i] for i in idx])  # [m, L, 8]
            for li in range(len(layers)):
                counts = np.bincount(r[:, li].ravel(), minlength=256).astype(np.int64)
                hs, cs, _ = pg.assign(counts, primary[li], secondary[li], hot[li], cost, pair,
                                      delta, target)
                H, C = int(hs.sum()), int(cs.sum())
                for k in full:
                    b = fit[k]
                    ep[k[0]] += max(b[0] + b[1] * hs[q] + b[2] * cs[q] for q in range(4))
                    ep[k[0] + " (mean GPU)"] += np.mean([b[0] + b[1] * hs[q] + b[2] * cs[q]
                                                         for q in range(4)])
                for k in sl:
                    b = fit[k]
                    tp[k[0]] += b[0] + b[1] * H + b[2] * C
        print(f"\nM={m}: MoE ms per step over {len(layers)} layers ({a.steps} held-out steps)")
        for name, v in sorted(ep.items()):
            print(f"  EP {name:24s} {v / a.steps / 1e3:6.2f}")
        for name, v in sorted(tp.items()):
            print(f"  TP-sliced {name:17s} {v / a.steps / 1e3:6.2f}")


if __name__ == "__main__":
    main()
