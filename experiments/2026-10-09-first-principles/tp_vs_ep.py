"""MoE time per 8-token step: expert-parallel as served vs every expert
sliced 4-way on its intermediate dim (each GPU reads 1/4 of every touched
expert; the partial outputs are summed by the all-reduce that already follows
the MoE).

EP: the shipped COST_US table (tiered_decode.cu) and the deployed replica
balancer (profile_grid.assign), slowest GPU per layer. TP-sliced: the
bench fit (../2026-10-08-m32, M=8: 22.8 + 5.67 h + 20.58 c us) on H/4 hot and
C/4 cold expert-equivalents, plus s us per extra expert slice. Agentic
held-out steps, prod profile at 3,670 hot.

    tp_vs_ep.py [--steps 4000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP / "2026-09-26-mimo-routing-profile"))
sys.path.insert(0, str(EXP / "2026-10-09-expert-coupling"))
from coupling import PROFILE, pg, runtime_hot  # noqa: E402
from mimo_replicas import load_steps  # noqa: E402

AGENTIC = Path("/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    _, _, _, _, cost, pair, delta, target = pg.configuration()
    prof = json.loads(PROFILE.read_text())
    layers = prof["routed_layers"]
    primary = np.asarray(prof["owners"], dtype=np.int32)
    secondary = np.asarray(prof["secondary_ranks"], dtype=np.int32)
    hot = runtime_hot(prof, 3670)
    steps = load_steps(AGENTIC, "heldout", layers)
    idx = np.random.default_rng(0).choice(len(steps), a.steps, replace=False)
    ep_max = ep_mean = 0.0
    H = C = 0.0
    hs_tot = np.zeros(4)
    for i in idx:
        for li in range(len(layers)):
            counts = np.bincount(steps[i][:, li].ravel(), minlength=256).astype(np.int64)
            hs, cs, _ = pg.assign(counts, primary[li], secondary[li], hot[li], cost, pair,
                                  delta, target)
            per = np.array([cost[hs[r], cs[r]] for r in range(4)], float)
            ep_max += per.max()
            ep_mean += per.mean()
            H += hs.sum()
            C += cs.sum()
    n = a.steps
    print(f"per step (75 MoE layers), {n} held-out steps:")
    print(f"  EP slowest GPU (as served)  {ep_max / n / 1e3:6.2f} ms")
    print(f"  EP mean GPU                 {ep_mean / n / 1e3:6.2f} ms")
    print(f"  touched per layer: hot {H / n / 75:.1f}, cold {C / n / 75:.2f} (all GPUs)")
    for s in (0.0, 0.5, 1.0, 2.0):
        h, c = H / n / 75, C / n / 75
        per_layer = 22.8 + 5.67 * h / 4 + 20.58 * c / 4 + s * (h + c) * 3 / 4
        print(f"  TP-sliced, +{s:.1f} us/extra slice: {per_layer * 75 / 1e3:6.2f} ms")


if __name__ == "__main__":
    main()
