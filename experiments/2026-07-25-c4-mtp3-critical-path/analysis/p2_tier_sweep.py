#!/usr/bin/env python3
"""P2: sweep the hot-slot budget to find the tier balance that minimises the
routed span at q16 decode.

The span model is validated against the MTP3 c4 trace to ~1% (25.96 predicted
vs 25.72 measured), so this sweep is a legitimate offline stand-in for a
cluster run. Fewer hot slots also releases HBM, which is the constraint on
concurrency.
"""
import argparse, json
from pathlib import Path
import numpy as np

import p1_balance as P

EXPERT_BYTES = 19_464_192


def hot_by_budget(prob, owners, slots_per_rank):
    """Pick the highest-activation experts per rank until the budget is spent."""
    hot = np.zeros_like(owners, dtype=bool)
    for r in range(P.EP):
        ly, ex = np.nonzero(owners == r)
        order = np.argsort(-prob[ly, ex], kind="stable")[:slots_per_rank]
        hot[ly[order], ex[order]] = True
    return hot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    reqs = P.load_routes(args.trace_dir)
    prof = json.loads(args.profile.read_text())
    owners = np.array(prof["owners"], dtype=np.int16)
    shipped_hot = np.zeros((len(P.ROUTED), P.NUM_EXPERTS), dtype=bool)
    for i, ids in enumerate(prof["hot_experts"]):
        shipped_hot[i, np.array(ids, dtype=int)] = True

    prob = P.activation_prob([r for r in reqs if r[0]["split"] == "train"])
    steps = P.c4_steps(reqs, np.random.default_rng(0), args.steps)

    def span_of(hot):
        cs = np.stack([P.layer_rank_cost(s, owners, hot) for s in steps])
        per_layer = cs.max(axis=2)
        # tier occupancy, for interpretation
        nh = nc = 0.0
        for s in steps:
            for ly in range(s.shape[1]):
                ids = np.unique(s[:, ly, :])
                own = owners[ly, ids]
                for r in range(P.EP):
                    sel = own == r
                    nh += np.count_nonzero(sel & hot[ly, ids])
                    nc += np.count_nonzero(sel & ~hot[ly, ids])
        n = len(steps) * s.shape[1] * P.EP
        return per_layer.sum(axis=1).mean() / 1000, nh / n, nc / n

    base, bh, bc = span_of(shipped_hot)
    print(f"shipped profile: {int(shipped_hot.sum())//P.EP} hot slots/rank nominal, "
          f"span {base:.3f} ms  (measured 25.72)")
    print(f"  activated per layer per rank: hot {bh:.2f}  cold {bc:.2f}\n")

    total = len(P.ROUTED) * P.PER_RANK
    print(f"{'hot/rank':>9} {'HBM/rank':>10} {'span ms':>9} {'vs base':>9} "
          f"{'act hot':>8} {'act cold':>9} {'t_hot':>8} {'t_cold':>8}")
    best = None
    for slots in (3300, 3023, 2870, 2600, 2300, 2000, 1800, 1600, 1400, 1200):
        hot = hot_by_budget(prob, owners, min(slots, total))
        sp, ah, ac = span_of(hot)
        gb = slots * EXPERT_BYTES / 1e9
        th, tc = ah * P.US_HOT * 75 / 1000, ac * P.US_COLD * 75 / 1000
        mark = ""
        if best is None or sp < best[1]:
            best = (slots, sp); mark = ""
        print(f"{slots:9d} {gb:9.1f}G {sp:9.3f} {sp-base:+9.3f} {ah:8.2f} {ac:9.2f} "
              f"{th:8.2f} {tc:8.2f}{mark}")

    slots, sp = best
    print(f"\nminimum at {slots} hot slots/rank: {sp:.3f} ms "
          f"({sp-base:+.3f} ms vs shipped, {100*(base-sp)/base:.1f}% of routed span)")
    freed = (3023 - slots) * EXPERT_BYTES / 1e9
    print(f"HBM released vs the 7 GB-reserve run (3023 hot/rank): {freed:.1f} GB/rank")


if __name__ == "__main__":
    main()
