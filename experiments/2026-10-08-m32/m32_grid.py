"""Hot / cold experts per GPU per MoE layer at 8, 16 and 32 tokens per step.

Live Claude Code capture (8-token verify steps per file). A 16- or 32-token
step is 2 or 4 concurrent requests: 8-token steps from different files,
paired at random. Prod placement: the frequency-promoted profile at 3,670 hot
per GPU, its replicas, and the deployed balancer (profile_grid.assign) with
the shipped cost table extended linearly past its grid.

Prints, per M: the (hot, cold) cells covering 95% of rank-layer calls (most
frequent first), the same for the slowest rank of each layer (the one the
all-reduce waits for), quantiles, and how often an expert gets > 8 tokens.

    m32_grid.py [--steps 6000] [--out grid.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "2026-10-04-tiered-decode-kernel-review"))
import profile_grid as pg  # noqa: E402

runtime_hot = pg.load("plot_glm_m32", EXP / "2026-09-27-glm53-mtp7-profile/plot_glm.py").runtime_hot

PROFILE = EXP.parent / "profiles/glm53-w4a16-agentic-3239-r2000-ccfreq3676.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--slots", type=int, default=3670)
    ap.add_argument("--out")
    a = ap.parse_args()
    _, _, _, _, cost, pair, delta, target = pg.configuration()
    profile = json.loads(PROFILE.read_text())
    primary = np.asarray(profile["owners"], dtype=np.int32)
    secondary = np.asarray(profile["secondary_ranks"], dtype=np.int32)
    hot = runtime_hot(profile, a.slots)
    src = pg.SOURCES[0]
    files = [json.loads(l)["file"] for l in (src / "manifest.jsonl").read_text().splitlines()]
    steps = []  # (file index, [8, 78, 8] routes)
    for i, f in enumerate(files):
        raw = np.load(src / f)
        r = raw.reshape(-1, 8, 78, 8)
        steps += [(i, r[j]) for j in range(r.shape[0])]
    print(f"{len(files)} files, {len(steps)} 8-token steps, hot/GPU "
          f"{hot.sum() / 4:.0f}", flush=True)
    rng = np.random.default_rng(0)
    out = {}
    for m in (8, 16, 32):
        reqs = m // 8
        hist = np.zeros((65, 65), np.int64)
        crit = np.zeros_like(hist)
        ntok = np.zeros(33, np.int64)
        for _ in range(a.steps):
            # reqs 8-token steps from distinct files
            picks = []
            while len(picks) < reqs:
                fi, r = steps[rng.integers(len(steps))]
                if all(fi != p[0] for p in picks):
                    picks.append((fi, r))
            routes = np.concatenate([r for _, r in picks])  # [m, 78, 8]
            for li in range(75):
                counts = np.bincount(routes[:, li + 3].ravel(), minlength=256).astype(np.int64)
                hs, cs, _ = pg.assign(counts, primary[li], secondary[li], hot[li], cost,
                                      pair, delta, target)
                ntok += np.bincount(counts[counts > 0], minlength=33)[:33]
                worst = max(range(4), key=lambda q: cost[hs[q], cs[q]])
                for q in range(4):
                    hist[hs[q], cs[q]] += 1
                crit[hs[worst], cs[worst]] += 1
        res = {}
        for name, h in (("all ranks", hist), ("slowest rank", crit)):
            flat = [(int(h[i, j]), i, j) for i in range(65) for j in range(65) if h[i, j]]
            flat.sort(reverse=True)
            tot = sum(v for v, _, _ in flat)
            cells, acc = [], 0
            for v, i, j in flat:
                cells.append((i, j))
                acc += v
                if acc >= 0.95 * tot:
                    break
            hh = np.repeat(np.arange(65), h.sum(1))
            cc = np.repeat(np.arange(65), h.sum(0))
            q = lambda x, p: int(np.percentile(x, p))  # noqa: E731
            print(f"M={m:2d} {name:12s}: {len(cells)} cells cover 95%; hot p5/p50/p95/max "
                  f"{q(hh, 5)}/{q(hh, 50)}/{q(hh, 95)}/{hh.max()}, cold "
                  f"{q(cc, 5)}/{q(cc, 50)}/{q(cc, 95)}/{cc.max()}; box hot "
                  f"{min(c[0] for c in cells)}-{max(c[0] for c in cells)} cold "
                  f"{min(c[1] for c in cells)}-{max(c[1] for c in cells)}", flush=True)
            res[name] = {"cells": cells, "hist": h.tolist()}
        big = ntok[9:].sum() / ntok.sum()
        print(f"M={m:2d} experts by tokens routed: "
              + " ".join(f"{k}:{ntok[k] / ntok.sum():.3f}" for k in range(1, 9))
              + f" >8:{big:.4f} max {np.flatnonzero(ntok).max()}", flush=True)
        res["ntok"] = ntok.tolist()
        out[m] = res
    if a.out:
        Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
