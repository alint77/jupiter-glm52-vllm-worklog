"""Sliced bench cells at M tokens: per layer and step, (distinct hot, distinct
cold) experts touched on the live Claude Code routing capture, M/8 8-token
steps from distinct files, hot = each layer's top --hot by frequency (as
cold_touch.py). Prints quantiles and the binned cells covering 95% of calls.

    sl_grid.py --m 32,64 [--hot 181] [--steps 400] [--hbin 12] [--cbin 4]
"""
import argparse
import json
from collections import Counter

import numpy as np

import cold_touch as ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="32,64")
    ap.add_argument("--hot", type=int, default=181)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--hbin", type=int, default=12)
    ap.add_argument("--cbin", type=int, default=4)
    a = ap.parse_args()
    src = ct.pg.SOURCES[0]
    files = [json.loads(l)["file"] for l in (src / "manifest.jsonl").read_text().splitlines()]
    steps = []
    for i, f in enumerate(files):
        r = np.load(src / f).reshape(-1, 8, 78, 8)
        steps += [(i, r[j]) for j in range(r.shape[0])]
    allr = np.stack([r for _, r in steps])
    freq = np.stack([np.bincount(allr[:, :, li + 3].ravel(), minlength=256) for li in range(75)])
    hot = np.zeros((75, 256), bool)
    for li in range(75):
        hot[li, np.argsort(-freq[li])[:a.hot]] = True
    rng = np.random.default_rng(0)
    for m in map(int, a.m.split(",")):
        k = m // 8
        hs, cs, maxtok = [], [], []
        for _ in range(a.steps):
            picks, seen = [], set()
            while len(picks) < k:
                fi, r = steps[rng.integers(len(steps))]
                if fi not in seen:
                    seen.add(fi)
                    picks.append(r)
            routes = np.concatenate(picks)
            for li in range(75):
                cnt = np.bincount(routes[:, li + 3].ravel(), minlength=256)
                hs.append(((cnt > 0) & hot[li]).sum())
                cs.append(((cnt > 0) & ~hot[li]).sum())
                maxtok.append(cnt.max())
        hs, cs, mt = map(np.array, (hs, cs, maxtok))
        q = lambda x: "/".join(f"{np.percentile(x, p):.0f}" for p in (5, 50, 95)) + f"/{x.max()}"
        print(f"M={m}: hot p5/p50/p95/max {q(hs)}  cold {q(cs)}  tokens/expert max p50/p95 "
              f"{np.percentile(mt, 50):.0f}/{np.percentile(mt, 95):.0f}  >8 tokens in "
              f"{(mt > 8).mean():.1%} of layer calls")
        cells = Counter(zip((hs + a.hbin // 2) // a.hbin * a.hbin, (cs + a.cbin // 2) // a.cbin * a.cbin))
        tot, acc, keep = sum(cells.values()), 0, []
        for c, n in cells.most_common():
            keep.append(c)
            acc += n
            if acc >= 0.95 * tot:
                break
        print(f"  {len(keep)} cells cover 95%: " + " ".join(f"{h},{c}" for h, c in sorted(keep)))


if __name__ == "__main__":
    main()
