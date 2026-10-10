"""Mean (hot, cold) distinct experts touched per sliced layer call for MTP3
batches of n sequences (4 tokens each, from distinct capture files), hot =
each layer's top --hot experts by frequency (as sl_grid.py).

    sl_mix.py [--n 1,4,8,16] [--hot 181] [--steps 1000]
"""
import argparse
import json

import numpy as np

import cold_touch as ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1,4,8,16")
    ap.add_argument("--hot", type=int, default=181)
    ap.add_argument("--steps", type=int, default=1000)
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
    for n in map(int, a.n.split(",")):
        hs, cs = [], []
        for _ in range(a.steps):
            picks, seen = [], set()
            while len(picks) < n:
                fi, r = steps[rng.integers(len(steps))]
                if fi not in seen:
                    seen.add(fi)
                    picks.append(r[:4])
            routes = np.concatenate(picks)
            for li in range(75):
                cnt = np.bincount(routes[:, li + 3].ravel(), minlength=256)
                hs.append(((cnt > 0) & hot[li]).sum())
                cs.append(((cnt > 0) & ~hot[li]).sum())
        hs, cs = np.array(hs), np.array(cs)
        print(json.dumps({"n": n, "m": 4 * n, "hot_mean": round(hs.mean(), 2), "cold_mean": round(cs.mean(), 2),
                          "cold_zero_frac": round((cs == 0).mean(), 3),
                          "hot_p50": float(np.median(hs)), "cold_p50": float(np.median(cs))}))


if __name__ == "__main__":
    main()
