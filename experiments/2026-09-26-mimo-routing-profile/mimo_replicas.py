#!/usr/bin/env python3
"""Choose Grace replica copies for MiMo-V2.6 and replay their effect exactly.

The runtime (`tiered_moe_scheduler`) assigns each active cold expert of a
decode step to its primary or its replica holder so that the largest number
of active cold experts on any rank is minimal. For four ranks that optimum has
a closed form (Hakimi's orientation theorem): with unreplicated cold experts as
self-loops and replicated ones as edges between their two holders, the minimum
achievable maximum load is ``max over rank subsets S of ceil(e(S) / |S|)``,
where e(S) counts loops and edges with both ends in S. That makes the replay
exact and vectorised over every held-out verify step.

Replica choice is greedy on the training split: a copy of expert e on rank d is
worth, per step, how much it lowers that layer's optimum when added alone to
the no-replica placement. Copies are taken best-first under a per-rank budget,
one copy per expert. Evaluation is on held-out task families only.

    mimo_replicas.py --trace-dir DIR --profile P.json --budgets 500 1000 ...
        [--write-profile BUDGET OUT.json]
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

EP = 4
STEP = 8
SUBSETS = [s for k in range(1, EP + 1) for s in itertools.combinations(range(EP), k)]
PAIRS = [(a, b) for a in range(EP) for b in range(a + 1, EP)]


def load_steps(trace_dir: Path, split: str, layers: list[int]) -> np.ndarray:
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    out = []
    for record in manifest:
        if record["split"] != split:
            continue
        routes = np.load(trace_dir / record["file"])[:, layers, :]
        usable = routes.shape[0] // STEP * STEP
        out.append(routes[:usable].reshape(-1, STEP, len(layers), routes.shape[2]))
    return np.concatenate(out)


def active_mask(steps: np.ndarray, num_experts: int) -> np.ndarray:
    n, _, layers, _ = steps.shape
    present = np.zeros((n, layers, num_experts), dtype=bool)
    for layer in range(layers):
        np.put_along_axis(present[:, layer], steps[:, :, layer, :].reshape(n, -1), True, 1)
    return present


def optimum(loops: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Min-max load. loops [..., EP], edges [..., 6] in PAIRS order."""
    best = np.zeros(loops.shape[:-1], dtype=np.int64)
    for subset in SUBSETS:
        inside = loops[..., list(subset)].sum(-1)
        for index, (a, b) in enumerate(PAIRS):
            if a in subset and b in subset:
                inside = inside + edges[..., index]
        best = np.maximum(best, -(-inside // len(subset)))
    return best


def layer_problems(active: np.ndarray, owners: np.ndarray, hot: np.ndarray,
                   secondary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cold = active & ~hot[None]
    fixed = cold & (secondary[None] < 0)
    loops = np.stack([(fixed & (owners[None] == r)).sum(-1) for r in range(EP)], -1)
    low = np.minimum(owners, secondary)
    high = np.maximum(owners, secondary)
    edges = np.stack([(cold & (low[None] == a) & (high[None] == b)).sum(-1)
                      for a, b in PAIRS], -1)
    return loops, edges


def score(active: np.ndarray, owners: np.ndarray, hot: np.ndarray) -> np.ndarray:
    """gain[layer, expert, dest]: summed drop in the layer optimum, alone."""
    n_layers, n_experts = owners.shape
    none = np.full_like(owners, -1)
    loops, edges = layer_problems(active, owners, hot, none)
    base = optimum(loops, edges)  # [steps, layers]
    gain = np.zeros((n_layers, n_experts, EP))
    cold_active = active & ~hot[None]
    for dest in range(EP):
        # Moving one self-loop of rank r into an r-dest edge.
        for r in range(EP):
            if r == dest:
                continue
            new_loops = loops.copy()
            new_loops[..., r] -= 1
            new_edges = edges.copy()
            new_edges[..., PAIRS.index((min(r, dest), max(r, dest)))] += 1
            drop = base - optimum(new_loops, new_edges)  # [steps, layers]
            owned = cold_active & (owners[None] == r)  # [steps, layers, experts]
            gain[:, :, dest] += np.einsum("sl,sle->le", drop, owned)
    return gain


def place(gain: np.ndarray, owners: np.ndarray, budget: int) -> np.ndarray:
    order = np.argsort(-gain, axis=None, kind="stable")
    secondary = np.full(owners.shape, -1, dtype=np.int64)
    used = np.zeros(EP, dtype=np.int64)
    for flat in order:
        layer, expert, dest = np.unravel_index(flat, gain.shape)
        if gain[layer, expert, dest] <= 0:
            break
        if dest == owners[layer, expert] or secondary[layer, expert] >= 0:
            continue
        if used[dest] >= budget:
            continue
        secondary[layer, expert] = dest
        used[dest] += 1
        if (used >= budget).all():
            break
    return secondary


def evaluate(active, owners, hot, secondary) -> dict:
    loops, edges = layer_problems(active, owners, hot, secondary)
    best = optimum(loops, edges)
    per_rank = loops + 0  # mean cold load is unchanged by orientation
    total = loops.sum(-1) + edges.sum(-1)
    return {
        "critical_cold_per_step": float(best.sum(1).mean()),
        "mean_cold_per_rank_per_step": float((total / EP).sum(1).mean()),
        "slack_per_step": float((best - total / EP).sum(1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--write-profile", nargs=2, metavar=("BUDGET", "OUT"))
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    layers = profile["routed_layers"]
    owners = np.asarray(profile["owners"])
    hot = np.zeros(owners.shape, dtype=bool)
    for layer, ids in enumerate(profile["hot_experts"]):
        hot[layer, ids] = True
    num_experts = owners.shape[1]
    train = active_mask(load_steps(args.trace_dir, "train", layers), num_experts)
    held = active_mask(load_steps(args.trace_dir, "heldout", layers), num_experts)
    gain = score(train, owners, hot)
    results = {"steps": {"train": len(train), "heldout": len(held)}}
    results["none"] = evaluate(held, owners, hot, np.full(owners.shape, -1))
    placements = {}
    for budget in args.budgets:
        secondary = place(gain, owners, budget)
        placements[budget] = secondary
        results[str(budget)] = evaluate(held, owners, hot, secondary) | {
            "replicas_per_rank": np.bincount(secondary[secondary >= 0], minlength=EP).tolist()
        }
    print(json.dumps(results, indent=2))
    if args.write_profile:
        budget, out = int(args.write_profile[0]), Path(args.write_profile[1])
        written = dict(profile, profile_version=2,
                       secondary_ranks=placements[budget].tolist(),
                       optimizer=profile["optimizer"] + f"+replicas-minmax-cold-{budget}")
        out.write_text(json.dumps(written) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
