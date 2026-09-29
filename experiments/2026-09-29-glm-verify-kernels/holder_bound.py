#!/usr/bin/env python3
"""How much can more holders per cold expert balance the MoE? Offline, on the
held-out split of the captured agentic routes.

Per step and layer, each active cold expert must be read by one of its holders
(owner, plus the profile's secondary); the busiest GPU's cold count, minimised
over those choices, is max over GPU subsets U of ceil(#experts whose holders
all lie in U / |U|) (Hall's condition; the same bound mimo_replicas.optimum
computes for pairs). Compared with every cold expert allowed on all 4 GPUs
(the ceiling of any extra-holder scheme) and with the pinned hot experts'
per-GPU counts, which no replica can move.

    holder_bound.py --trace-dir MERGED --profile PROFILE.json
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-routing-profile"))
from mimo_replicas import active_mask, load_steps  # noqa: E402

EP = 4
SUBSETS = [s for k in range(1, EP + 1) for s in itertools.combinations(range(EP), k)]


def minmax(counts: np.ndarray) -> np.ndarray:
    """counts [..., 16] per holder bitmask -> min-max load [...]."""
    best = np.zeros(counts.shape[:-1], dtype=np.int64)
    for subset in SUBSETS:
        u = sum(1 << r for r in subset)
        inside = sum(counts[..., m] for m in range(1, 16) if m & ~u == 0)
        best = np.maximum(best, -(-inside // len(subset)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    args = ap.parse_args()
    p = json.loads(args.profile.read_text())
    owners = np.asarray(p["owners"])
    secondary = np.asarray(p.get("secondary_ranks", np.full(owners.shape, -1)))
    hot = np.zeros(owners.shape, dtype=bool)
    for layer, ids in enumerate(p["hot_experts"]):
        hot[layer, ids] = True
    active = active_mask(load_steps(args.trace_dir, "heldout", p["routed_layers"]),
                         owners.shape[1])  # [steps, layers, experts]
    cold = active & ~hot[None]
    mask = (1 << owners) | np.where(secondary >= 0, 1 << np.maximum(secondary, 0), 0)
    counts = np.stack([(cold & (mask[None] == m)).sum(-1) for m in range(16)], -1)
    now = minmax(counts)  # [steps, layers]
    total = cold.sum(-1)
    flex = -(-total // EP)  # every cold expert on any GPU
    per_rank_hot = np.stack([(active & hot[None] & (owners[None] == r)).sum(-1)
                             for r in range(EP)], -1)
    per_rank_cold_fixed = np.stack([(cold & (owners[None] == r)).sum(-1)
                                    for r in range(EP)], -1)
    two_holder = (secondary >= 0) & ~hot
    n = len(cold)
    per_step = lambda x: float(x.sum(1).mean())  # summed over layers, mean over steps
    print(f"held-out steps {n}, layers {cold.shape[1]}; cold experts with a second "
          f"holder {int(two_holder.sum())} of {int((~hot).sum())}")
    print(f"cold per step (sum over layers): mean/GPU {per_step(total / EP):.1f}")
    print(f"  busiest GPU, owner only          {per_step(per_rank_cold_fixed.max(-1)):.1f}")
    print(f"  busiest GPU, owner + secondary   {per_step(now):.1f}   <- served")
    print(f"  busiest GPU, any GPU (ceiling)   {per_step(flex):.1f}")
    print(f"hot per step (sum over layers): mean/GPU {per_step(per_rank_hot.mean(-1)):.1f}, "
          f"busiest GPU {per_step(per_rank_hot.max(-1)):.1f}")
    both = np.stack([now, flex], -1)
    frac = (now > flex).mean()
    print(f"layer-steps where a third/fourth holder could lower the busiest cold "
          f"count: {frac:.1%}; mean excess there {float((now - flex)[now > flex].mean()) if frac else 0:.2f}")


if __name__ == "__main__":
    main()
