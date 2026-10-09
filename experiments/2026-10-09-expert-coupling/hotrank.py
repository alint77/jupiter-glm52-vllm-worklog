"""Hot-set ranking: per-route frequency vs per-step touch frequency.

Within one request's step the 8 tokens' routes cluster (an expert routed
twice in a step costs one load), so an expert's route count overstates how
often it has to be loaded. Re-pick each (layer, owner GPU)'s hot experts --
same owners, same hot count per (layer, GPU) as the prod profile at 3,381/GPU
-- by how many training steps touch the expert at M tokens, and count cold
experts per GPU-layer on held-out requests at M = 8 and 32.

    hotrank.py [--eval-steps 3000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from coupling import NL, PROFILE, E, load, runtime_hot  # noqa: E402


def steps_of(reqs, m, n, rng):
    out = []
    while len(out) < n:
        fis = rng.choice(len(reqs), m // 8, replace=False)
        out.append(np.concatenate([reqs[f][rng.integers(len(reqs[f]))] for f in fis]))
    return out


def touch(steps):
    t = np.zeros((NL, E))
    for r in steps:
        for li in range(NL):
            t[li, np.unique(r[:, li])] += 1
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-steps", type=int, default=3000)
    a = ap.parse_args()
    reqs = load()
    train, test = reqs[0::2], reqs[1::2]
    prof = json.loads(PROFILE.read_text())
    owners = np.asarray(prof["owners"])
    hot0 = runtime_hot(prof, 3381)
    rng = np.random.default_rng(0)
    tr = np.concatenate([r.reshape(-1, NL, 8) for r in train])
    route = np.stack([np.bincount(tr[:, li].ravel(), minlength=E) for li in range(NL)])
    metrics = {"route frequency (train)": route}
    for m in (8, 32):
        metrics[f"step touch at M={m} (train)"] = touch(steps_of(train, m, 20000, rng))

    def pick(score):
        hot = np.zeros_like(hot0)
        for li in range(NL):
            for g in range(4):
                own = np.flatnonzero(owners[li] == g)
                k = hot0[li, own].sum()
                hot[li, own[np.argsort(score[li, own])[::-1][:k]]] = True
        return hot

    def pick_global(score):
        """Each GPU's hot count spread over layers by the metric, as prod's
        promotion does."""
        hot = np.zeros_like(hot0)
        for g in range(4):
            mine = owners == g
            k = hot0[mine].sum()
            s = np.where(mine, score, -1.0)
            hot.flat[np.argsort(s, axis=None)[::-1][:k]] = True
        return hot

    sets = {"prod profile": hot0} | {k: pick(v) for k, v in metrics.items()}
    sets |= {k.replace("(train)", "(train, over layers)"): pick_global(v)
             for k, v in metrics.items()}
    for m in (8, 32):
        ev = steps_of(test, m, a.eval_steps, np.random.default_rng(1))
        print(f"\nM={m}, held-out: cold experts per GPU-layer (mean / slowest GPU of the layer)")
        for name, hot in sets.items():
            cold = np.zeros((len(ev), NL, 4))
            for i, r in enumerate(ev):
                for li in range(NL):
                    u = np.unique(r[:, li])
                    u = u[~hot[li, u]]
                    cold[i, li] = np.bincount(owners[li, u], minlength=4)
            print(f"  {name:42s} {cold.mean():.3f} / {cold.max(2).mean():.3f}")


if __name__ == "__main__":
    main()
