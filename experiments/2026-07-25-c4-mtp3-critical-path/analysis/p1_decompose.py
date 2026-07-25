#!/usr/bin/env python3
"""Split the per-layer EP imbalance into expectation vs per-step variance.

Only the expectation component is reachable by a static owner map. The variance
component is the max-of-4-ranks order statistic of which experts happen to fire
in a given step, and no fixed assignment removes it.
"""
import argparse, json
from pathlib import Path
import numpy as np

from p1_balance import (ROUTED, EP, NUM_EXPERTS, PER_RANK, US_HOT, US_COLD,
                        load_routes, c4_steps, layer_rank_cost, activation_prob,
                        balance_owners)


def decompose(steps, owners, hot_mask):
    cs = np.stack([layer_rank_cost(s, owners, hot_mask) for s in steps])  # (S,L,R)
    realized = (cs.max(axis=2) - cs.mean(axis=2)).sum(axis=1).mean()
    E = cs.mean(axis=0)                                                   # (L,R)
    expectation = (E.max(axis=1) - E.mean(axis=1)).sum()
    return realized / 1000, expectation / 1000, cs


def hot_slots_per_rank(owners, hot_mask):
    return [int((hot_mask & (owners == r)).sum()) for r in range(EP)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    reqs = load_routes(args.trace_dir)
    prof = json.loads(args.profile.read_text())
    owners = np.array(prof["owners"], dtype=np.int16)
    hot_mask = np.zeros((len(ROUTED), NUM_EXPERTS), dtype=bool)
    for i, ids in enumerate(prof["hot_experts"]):
        hot_mask[i, np.array(ids, dtype=int)] = True

    rng = np.random.default_rng(0)
    steps = c4_steps(reqs, rng, args.steps)

    real, exp, cs = decompose(steps, owners, hot_mask)
    print(f"=== imbalance decomposition, {args.steps} simulated c4 steps ===")
    print(f"  realized per-step imbalance (max-mean)   {real:7.3f} ms/step")
    print(f"  expectation component (statically fixable){exp:7.3f} ms/step "
          f"({100*exp/real:.0f}%)")
    print(f"  variance component (order statistic)     {real-exp:7.3f} ms/step "
          f"({100*(real-exp)/real:.0f}%)")

    print("\n  small-sample bias check: imbalance measured from N steps")
    for n in (5, 10, 20, 50, 100, args.steps):
        sub = cs[:n]
        E = sub.mean(axis=0)
        print(f"    N={n:4d}: apparent 'persistent' component "
              f"{(E.max(axis=1)-E.mean(axis=1)).sum()/1000:6.3f} ms")

    print(f"\n  hot slots per rank (shipped): {hot_slots_per_rank(owners, hot_mask)}")

    prob = activation_prob([r for r in reqs if r[0]["split"] == "train"])
    new_owners = balance_owners(prob, hot_mask)
    real2, exp2, _ = decompose(steps, new_owners, hot_mask)
    print(f"\n=== after count-free expectation rebalancing ===")
    print(f"  realized    {real2:7.3f} ({real2-real:+.3f})")
    print(f"  expectation {exp2:7.3f} ({exp2-exp:+.3f})")
    print(f"  hot slots per rank: {hot_slots_per_rank(new_owners, hot_mask)}"
          f"   <- must stay balanced for the per-rank HBM budget")


if __name__ == "__main__":
    main()
