#!/usr/bin/env python3
"""P1: per-layer EP owner balancing against distinct-activated-expert cost.

The shipped owner map (`greedy-balanced-owner`) balances the *number of routing
assignments* per rank per layer. At q16 decode the routed Marlin cost is set by
the number of *distinct activated experts* (weight streaming), and a cold expert
costs ~2.7x a hot one. Those objectives are not the same, which is why the
c4 MTP3 trace still shows 7.76 ms/step of per-layer rank skew.

This script rebuilds the owner map against the correct objective and predicts
the change, entirely offline from the captured routing traces.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROUTED = tuple(range(3, 78))
NUM_EXPERTS = 256
EP = 4
PER_RANK = NUM_EXPERTS // EP

# Per activated expert per layer, derived in the critical-path review from the
# MTP3 c4 trace (s13_model.py): hot 1.28 ms / 75 layers, cold 3.47 ms / 75.
US_HOT = 1280.0 / 75
US_COLD = 3467.0 / 75


def load_routes(trace_dir: Path):
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    out = []
    for entry in manifest:
        z = np.load(trace_dir / entry["file"])
        # (steps, verify_tokens, layers, topk) -> routed layers only
        out.append((entry, z["routes"][:, :, ROUTED[0]:, :].astype(np.int16)))
    return out


def c4_steps(reqs, rng, n_steps):
    """Assemble concurrency-4 steps: 4 independent requests, one step each."""
    out = []
    for _ in range(n_steps):
        pick = rng.choice(len(reqs), size=EP, replace=False)
        toks = []
        for i in pick:
            r = reqs[i][1]
            toks.append(r[rng.integers(r.shape[0])])   # (verify, layers, topk)
        out.append(np.concatenate(toks, axis=0))        # (16, layers, topk)
    return out


def layer_rank_cost(step, owners, hot_mask):
    """Per-layer, per-rank routed span (us) for one c4 step."""
    nl = step.shape[1]
    cost = np.zeros((nl, EP))
    for ly in range(nl):
        ids = np.unique(step[:, ly, :])
        own = owners[ly, ids]
        hot = hot_mask[ly, ids]
        for r in range(EP):
            sel = own == r
            n_hot = int(np.count_nonzero(sel & hot))
            n_cold = int(np.count_nonzero(sel & ~hot))
            # hot and cold tiers overlap on separate streams
            cost[ly, r] = max(n_hot * US_HOT, n_cold * US_COLD)
    return cost


def evaluate(steps, owners, hot_mask):
    span = 0.0
    imbalance = 0.0
    for st in steps:
        c = layer_rank_cost(st, owners, hot_mask)
        span += c.max(axis=1).sum()
        imbalance += (c.max(axis=1) - c.mean(axis=1)).sum()
    return span / len(steps) / 1000, imbalance / len(steps) / 1000  # ms


def activation_prob(reqs, n_tokens=16):
    """P(expert activated somewhere in an n_tokens step), per layer/expert."""
    nl = len(ROUTED)
    hits = np.zeros((nl, NUM_EXPERTS))
    tot = 0
    for _, r in reqs:
        flat = r.reshape(-1, nl, r.shape[-1])       # (tokens, layers, topk)
        n = flat.shape[0] // n_tokens
        for w in range(n):
            blk = flat[w * n_tokens:(w + 1) * n_tokens]
            for ly in range(nl):
                hits[ly, np.unique(blk[:, ly, :])] += 1
        tot += n
    return hits / max(tot, 1)


def balance_owners(prob, hot_pref, cost_hot=US_HOT, cost_cold=US_COLD):
    """Greedy LPT on expected per-step activation cost, capacity-constrained."""
    owners = np.empty((len(ROUTED), NUM_EXPERTS), dtype=np.int16)
    for ly in range(len(ROUTED)):
        w = prob[ly] * np.where(hot_pref[ly], cost_hot, cost_cold)
        load = np.zeros(EP)
        size = np.zeros(EP, dtype=int)
        for e in np.argsort(-w, kind="stable"):
            r = min((r for r in range(EP) if size[r] < PER_RANK),
                    key=lambda c: (load[c], c))
            owners[ly, e] = r
            load[r] += w[e]
            size[r] += 1
    return owners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-profile", type=Path)
    args = ap.parse_args()

    reqs = load_routes(args.trace_dir)
    prof = json.loads(args.profile.read_text())
    assert prof["ep_size"] == EP and prof["num_experts"] == NUM_EXPERTS
    owners = np.array(prof["owners"], dtype=np.int16)
    hot_mask = np.zeros((len(ROUTED), NUM_EXPERTS), dtype=bool)
    for i, ids in enumerate(prof["hot_experts"]):
        hot_mask[i, np.array(ids, dtype=int)] = True

    train = [r for r in reqs if r[0]["split"] == "train"]
    held = [r for r in reqs if r[0]["split"] != "train"]
    print(f"requests: {len(train)} train, {len(held)} held-out")
    print(f"cost model: hot {US_HOT:.2f} us, cold {US_COLD:.2f} us per activated "
          f"expert per layer (ratio {US_COLD/US_HOT:.2f})\n")

    rng = np.random.default_rng(args.seed)
    ev_steps = c4_steps(reqs, rng, args.steps)

    span0, imb0 = evaluate(ev_steps, owners, hot_mask)
    print("=== shipped owner map (greedy-balanced on assignment counts) ===")
    print(f"  predicted routed span     {span0:7.3f} ms/step   (measured 25.72)")
    print(f"  predicted rank imbalance  {imb0:7.3f} ms/step   (measured  7.76)")

    prob = activation_prob(train)
    new_owners = balance_owners(prob, hot_mask)
    span1, imb1 = evaluate(ev_steps, new_owners, hot_mask)
    print("\n=== rebalanced on expected distinct-activation cost ===")
    print(f"  predicted routed span     {span1:7.3f} ms/step   ({span1-span0:+.3f})")
    print(f"  predicted rank imbalance  {imb1:7.3f} ms/step   ({imb1-imb0:+.3f})")

    hs = c4_steps(held, np.random.default_rng(args.seed + 1), args.steps)
    hspan0, himb0 = evaluate(hs, owners, hot_mask)
    hspan1, himb1 = evaluate(hs, new_owners, hot_mask)
    print("\n=== held-out requests only ===")
    print(f"  shipped     span {hspan0:7.3f}  imbalance {himb0:7.3f}")
    print(f"  rebalanced  span {hspan1:7.3f}  imbalance {himb1:7.3f}"
          f"   ({100*(hspan1-hspan0)/hspan0:+.1f}% span)")

    moved = int((owners != new_owners).sum())
    print(f"\n  experts reassigned: {moved} of {owners.size} "
          f"({100*moved/owners.size:.1f}%)")
    for r in range(EP):
        print(f"    rank{r}: {int((new_owners==r).sum())} slots "
              f"(was {int((owners==r).sum())})")

    if args.output_profile:
        prof["owners"] = new_owners.tolist()
        prof["optimizer"] = "activation-cost-balanced-owner-v1"
        args.output_profile.write_text(json.dumps(prof))
        print(f"\n  wrote {args.output_profile}")


if __name__ == "__main__":
    main()
