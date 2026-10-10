"""Napkin input for cold prefetch at decode: on the live Claude Code routing
capture (../2026-10-08-m32's source), M-token steps made of M/8 8-token steps
from distinct files; the sliced hot set taken as each layer's most frequent
--hot experts (13,614 slices / 75 layers = 181.5 per GPU); per layer, the
number of distinct cold experts the step touches. Cold bytes per GPU = that x
the slice size (5.1 MiB).

    cold_touch.py [--hot 181] [--steps 400]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "2026-10-04-tiered-decode-kernel-review"))
import profile_grid as pg  # noqa: E402

SLICE_MIB = 5.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", type=int, default=181)
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    src = pg.SOURCES[0]
    files = [json.loads(l)["file"] for l in (src / "manifest.jsonl").read_text().splitlines()]
    steps = []
    for i, f in enumerate(files):
        r = np.load(src / f).reshape(-1, 8, 78, 8)
        steps += [(i, r[j]) for j in range(r.shape[0])]
    allr = np.stack([r for _, r in steps])  # [S, 8, 78, 8]
    freq = np.stack([np.bincount(allr[:, :, li + 3].ravel(), minlength=256) for li in range(75)])
    hot = np.zeros((75, 256), bool)
    for li in range(75):
        hot[li, np.argsort(-freq[li])[:a.hot]] = True
    cov = (freq * hot).sum() / freq.sum()
    print(f"{len(steps)} 8-token steps; hot {a.hot}/256 per layer covers {cov:.1%} of routed picks")
    rng = np.random.default_rng(0)
    for m in (8, 32, 64, 128, 256, 512):
        k = m // 8
        cold_n, hot_n = [], []
        for _ in range(a.steps):
            picks, seen = [], set()
            while len(picks) < k:
                fi, r = steps[rng.integers(len(steps))]
                if fi not in seen or len(seen) >= len(files):
                    seen.add(fi)
                    picks.append(r)
            routes = np.concatenate(picks)  # [m, 78, 8]
            for li in range(75):
                touched = np.zeros(256, bool)
                touched[routes[:, li + 3].ravel()] = True
                cold_n.append((touched & ~hot[li]).sum())
                hot_n.append((touched & hot[li]).sum())
        c = np.array(cold_n)
        print(f"M={m:4d}: touched hot/layer {np.mean(hot_n):6.1f}  cold/layer mean {c.mean():5.1f} "
              f"p95 {np.percentile(c, 95):4.0f} of {256 - a.hot}  -> cold MiB/layer/GPU "
              f"{c.mean() * SLICE_MIB:6.1f} (all cold {(256 - a.hot) * SLICE_MIB:.0f}), "
              f"per step {c.mean() * SLICE_MIB * 75 / 1024:5.2f} GiB")


if __name__ == "__main__":
    main()
