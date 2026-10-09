"""Temporal locality of cold experts: an HBM LRU cache vs the static hot set.

Per (layer, owner GPU): demote the S least-touched hot experts (agentic train
touch counts) and give the slots to an LRU of cold experts that GPU's steps
used (an expert read at step t is cached from step t + 1). Held-out agentic
requests replayed in order (c=1); no replicas in either arm. Cold experts per
GPU-layer per step: mean and slowest GPU.

    cold_cache.py [--steps 8000]
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP / "2026-09-26-mimo-routing-profile"))
sys.path.insert(0, str(EXP / "2026-10-09-expert-coupling"))
from coupling import PROFILE, runtime_hot  # noqa: E402
from mimo_replicas import active_mask, load_steps  # noqa: E402

AGENTIC = Path("/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    a = ap.parse_args()
    prof = json.loads(PROFILE.read_text())
    layers = prof["routed_layers"]
    owners = np.asarray(prof["owners"])
    hot0 = runtime_hot(prof, 3670)
    touch = active_mask(load_steps(AGENTIC, "train", layers), 256).sum(0)
    held = active_mask(load_steps(AGENTIC, "heldout", layers)[: a.steps], 256)
    L = len(layers)
    for S in (0, 1, 2, 4, 8):
        hot = hot0.copy()
        for li in range(L):
            for g in range(4):
                own_hot = np.flatnonzero(hot0[li] & (owners[li] == g))
                hot[li, own_hot[np.argsort(touch[li, own_hot])[:S]]] = False
        caches = [[OrderedDict() for _ in range(4)] for _ in range(L)]
        tot = np.zeros(4)
        worst = 0.0
        hits = 0
        for t in range(len(held)):
            for li in range(L):
                act = np.flatnonzero(held[t, li] & ~hot[li])
                per = np.zeros(4)
                for e in act:
                    g = owners[li, e]
                    c = caches[li][g]
                    if e in c:
                        c.move_to_end(e)
                        hits += 1
                        continue
                    per[g] += 1
                if S:
                    for e in act:
                        c = caches[li][owners[li, e]]
                        c[e] = True
                        c.move_to_end(e)
                        while len(c) > S:
                            c.popitem(last=False)
                tot += per
                worst += per.max()
        n = len(held) * L
        print(f"S={S}: demoted {S * 4 * L} hot/GPU-set ({S * L}/GPU); cold per GPU-layer "
              f"mean {tot.sum() / 4 / n:.3f}, slowest {worst / n:.3f}; cache hits/step "
              f"{hits / len(held):.1f}", flush=True)


if __name__ == "__main__":
    main()
