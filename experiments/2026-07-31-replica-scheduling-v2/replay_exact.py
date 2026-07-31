#!/usr/bin/env python3
"""Phase 0b: replay held-out routes under the exact cold-only assignment.

Compares four assignment policies on captured routing traces:

  off          every route runs on its primary owner (today's behaviour)
  greedy_v1    the reverted runtime greedy, two-resource additive scoring
  exact_cold   hot experts pinned to their primary; active cold experts
               distributed by exact min-max cardinality balancing
  exact_hatch  exact_cold plus a hot->secondary escape hatch, taken only when
               it strictly lowers the modelled layer maximum

The v2 claim is that ``exact_cold`` retains the span reduction while holding
total Grace activations flat, which is the quantity that drove the +3.972 ms
Marlin counter-cost measured in the v1 paired trace.

``exact_cold`` needs no cost constants: its assignment minimises the maximum
per-rank count of active cold experts. The cost model only scores the result,
so every policy is reported under all three models for robustness.
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

EP = 4
NUM_EXPERTS = 256
NUM_LAYERS = 75
VERIFY_TOKENS = 4
ROUTED_START = 3

# v1's additive scalars (vllm tiered_moe_scheduler.py at replica-scheduling-archive),
# expressed per layer as oracle.py did.
LEGACY_HBM_TASK_US = 1280.0 / NUM_LAYERS
LEGACY_GRACE_TASK_US = 3467.0 / NUM_LAYERS
# Calibrated chain model (cost-calibration-1129855 / 1130891).
HBM_CHAIN_US = 167.0
GRACE_CHAIN_BASE_US = 24.0
GRACE_TASK_US = 46.0

SUBSETS = [
    [rank for rank in range(EP) if mask >> rank & 1] for mask in range(1, 1 << EP)
]
PAIRS = [(i, j) for i in range(EP) for j in range(i + 1, EP)]
PAIR_INDEX = {pair: index for index, pair in enumerate(PAIRS)}
# Pair classes fully inside each subset, precomputed for the Hall-type test.
SUBSET_PAIRS = [
    [PAIR_INDEX[pair] for pair in PAIRS if pair[0] in subset and pair[1] in subset]
    for subset in SUBSETS
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument(
        "--placement",
        type=Path,
        action="append",
        required=True,
        help=(
            "Placement profile(s). Version-2 profiles carry owners, the "
            "deployed hot set and secondary_ranks together, so each one fully "
            "describes a configuration. Repeatable."
        ),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help=(
            "Optional base profile overriding owners and hot residency, "
            "trimmed to --hot-slots-per-rank. Only needed for placements that "
            "do not carry their own deployed hot set."
        ),
    )
    parser.add_argument("--hot-slots-per-rank", type=int, default=2614)
    parser.add_argument("--heldout-c1-steps", type=int, default=300)
    parser.add_argument("--heldout-c4-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Trace loading (matches oracle.py so results are comparable to Phase 1)
# --------------------------------------------------------------------------


def load_requests(trace_dir: Path, split: str) -> list[np.ndarray]:
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    requests = []
    for record in manifest:
        if record["split"] != split:
            continue
        routes = np.load(trace_dir / record["file"], mmap_mode="r")
        if routes.ndim != 3 or routes.shape[1:] != (78, 8):
            raise ValueError(f"Unexpected route shape: {routes.shape}")
        if routes.shape[0] % VERIFY_TOKENS:
            raise ValueError(f"Partial verification step: {routes.shape}")
        requests.append(
            np.asarray(routes[:, ROUTED_START:, :]).reshape(
                -1, VERIFY_TOKENS, NUM_LAYERS, 8
            )
        )
    if not requests:
        raise ValueError(f"No {split} requests in {trace_dir}")
    return requests


def sample_steps(
    requests: list[np.ndarray],
    count: int,
    concurrency: int,
    rng: np.random.Generator,
) -> np.ndarray:
    steps = np.concatenate(requests)
    picks = rng.integers(len(steps), size=(count, concurrency))
    return np.concatenate([steps[picks[:, index]] for index in range(concurrency)], 1)


def route_counts(steps: np.ndarray) -> np.ndarray:
    counts = np.zeros((steps.shape[0], NUM_LAYERS, NUM_EXPERTS), dtype=np.int16)
    for step in range(steps.shape[0]):
        for layer in range(NUM_LAYERS):
            counts[step, layer] = np.bincount(
                steps[step, :, layer, :].reshape(-1), minlength=NUM_EXPERTS
            )
    return counts


def load_profile(path: Path, hot_slots_per_rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Load owners and HBM residency, trimmed to the deployed hot-slot budget."""
    profile = json.loads(path.read_text())
    if profile["ep_size"] != EP or profile["num_experts"] != NUM_EXPERTS:
        raise ValueError("The replay requires EP4 with 256 experts")
    owners = np.asarray(profile["owners"], dtype=np.int8)
    if owners.shape != (NUM_LAYERS, NUM_EXPERTS):
        raise ValueError(f"Unexpected owner shape: {owners.shape}")
    hot_lists = [list(experts) for experts in profile["hot_experts"]]
    hot = np.zeros_like(owners, dtype=bool)
    for layer, experts in enumerate(hot_lists):
        hot[layer, np.asarray(experts, dtype=np.int64)] = True

    for rank in range(EP):
        current = int(np.count_nonzero(hot & (owners == rank)))
        remaining = current - hot_slots_per_rank
        while remaining > 0:
            for layer, experts in enumerate(hot_lists):
                for index in range(len(experts) - 1, -1, -1):
                    expert = experts[index]
                    if owners[layer, expert] == rank:
                        experts.pop(index)
                        hot[layer, expert] = False
                        remaining -= 1
                        break
                if remaining <= 0:
                    break
        remaining = hot_slots_per_rank - current
        while remaining > 0:
            progressed = False
            for layer in range(NUM_LAYERS):
                for expert in np.flatnonzero(owners[layer] == rank):
                    if not hot[layer, expert]:
                        hot[layer, expert] = True
                        remaining -= 1
                        progressed = True
                        break
                if remaining <= 0:
                    break
            if not progressed:
                raise ValueError("HBM target exceeds locally owned expert slots")
    return owners, hot


def load_placement(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load owners, deployed hot residency and secondary ranks from a v2 profile."""
    profile = json.loads(path.read_text())
    if profile["ep_size"] != EP or profile["num_experts"] != NUM_EXPERTS:
        raise ValueError("The replay requires EP4 with 256 experts")
    owners = np.asarray(profile["owners"], dtype=np.int8)
    secondary = np.asarray(profile["secondary_ranks"], dtype=np.int8)
    if owners.shape != (NUM_LAYERS, NUM_EXPERTS):
        raise ValueError(f"Unexpected owner shape: {owners.shape}")
    if secondary.shape != (NUM_LAYERS, NUM_EXPERTS):
        raise ValueError(f"Unexpected secondary shape: {secondary.shape}")
    hot = np.zeros_like(owners, dtype=bool)
    for layer, experts in enumerate(profile["hot_experts"]):
        hot[layer, np.asarray(experts, dtype=np.int64)] = True
    return owners, hot, secondary


# --------------------------------------------------------------------------
# Cost models
# --------------------------------------------------------------------------


def rank_times(hbm: np.ndarray, grace: np.ndarray, model: str) -> np.ndarray:
    """Per-rank modelled layer time from active HBM and Grace expert counts."""
    if model == "legacy":
        return np.maximum(LEGACY_HBM_TASK_US * hbm, LEGACY_GRACE_TASK_US * grace)
    if model == "chain":
        hbm_time = np.where(hbm > 0, HBM_CHAIN_US, 0.0)
        grace_time = np.where(
            grace > 0,
            np.maximum(HBM_CHAIN_US, GRACE_CHAIN_BASE_US + GRACE_TASK_US * grace),
            0.0,
        )
        return np.maximum(hbm_time, grace_time)
    if model == "grace_only":
        # The ordinal model v2's assignment actually relies on: Grace dominates,
        # HBM is free. Included to show the assignment is not model-tuned.
        return GRACE_TASK_US * grace
    raise ValueError(f"Unknown cost model: {model}")


COST_MODELS = ("legacy", "chain", "grace_only")


# --------------------------------------------------------------------------
# Exact min-max orientation
# --------------------------------------------------------------------------


def hall_feasible(offsets: np.ndarray, pair_counts: np.ndarray, limit: int) -> bool:
    """Hall-type test: can every edge be oriented within capacity ``limit``?

    An orientation with in-degree at most ``cap_r`` exists iff for every subset
    A of ranks, the edges with *both* endpoints in A fit in A's total capacity;
    edges with an endpoint outside A can always be oriented outward.
    """
    caps = limit - offsets
    if np.any(caps < 0):
        return False
    for subset, pairs in zip(SUBSETS, SUBSET_PAIRS):
        if pair_counts[pairs].sum() > caps[subset].sum():
            return False
    return True


def flow_orient(
    offsets: np.ndarray, pair_counts: np.ndarray, limit: int
) -> np.ndarray | None:
    """Ground-truth orientation by max flow over the 12-node class network.

    Returns ``split[p]`` = how many edges of pair class ``p`` go to its lower
    rank, or None when ``limit`` is infeasible.
    """
    caps = limit - offsets
    if np.any(caps < 0):
        return None
    # Node ids: 0 source, 1..6 pair classes, 7..10 ranks, 11 sink.
    size = 2 + len(PAIRS) + EP
    capacity = np.zeros((size, size), dtype=np.int64)
    for index, (low, high) in enumerate(PAIRS):
        capacity[0, 1 + index] = pair_counts[index]
        capacity[1 + index, 1 + len(PAIRS) + low] = pair_counts[index]
        capacity[1 + index, 1 + len(PAIRS) + high] = pair_counts[index]
    for rank in range(EP):
        capacity[1 + len(PAIRS) + rank, size - 1] = caps[rank]

    residual = capacity.copy()
    sink = size - 1
    total = 0
    while True:
        parent = [-1] * size
        parent[0] = 0
        queue = [0]
        while queue and parent[sink] < 0:
            node = queue.pop(0)
            for nxt in range(size):
                if parent[nxt] < 0 and residual[node, nxt] > 0:
                    parent[nxt] = node
                    queue.append(nxt)
        if parent[sink] < 0:
            break
        node, bottleneck = sink, np.iinfo(np.int64).max
        while node:
            bottleneck = min(bottleneck, int(residual[parent[node], node]))
            node = parent[node]
        node = sink
        while node:
            residual[parent[node], node] -= bottleneck
            residual[node, parent[node]] += bottleneck
            node = parent[node]
        total += bottleneck

    if total != int(pair_counts.sum()):
        return None
    split = np.zeros(len(PAIRS), dtype=np.int64)
    for index, (low, _) in enumerate(PAIRS):
        column = 1 + len(PAIRS) + low
        split[index] = capacity[1 + index, column] - residual[1 + index, column]
    return split


def solve_min_max(
    offsets: np.ndarray, pair_counts: np.ndarray, cross_check: bool
) -> tuple[int, np.ndarray]:
    """Return the optimal maximum load and a per-pair-class split achieving it."""
    total = int(pair_counts.sum())
    lower = max(int(offsets.max()), -(-(int(offsets.sum()) + total) // EP))
    upper = int(offsets.max()) + total
    limit = lower
    while limit <= upper:
        split = flow_orient(offsets, pair_counts, limit)
        if split is not None:
            if cross_check and not hall_feasible(offsets, pair_counts, limit):
                raise AssertionError("Hall test rejects a flow-feasible limit")
            if cross_check and limit > lower:
                if hall_feasible(offsets, pair_counts, limit - 1):
                    raise AssertionError("Hall test accepts an infeasible limit")
            return limit, split
        if cross_check and hall_feasible(offsets, pair_counts, limit):
            raise AssertionError("Hall test accepts a flow-infeasible limit")
        limit += 1
    raise RuntimeError("No feasible orientation found")


# --------------------------------------------------------------------------
# Policies. Each returns per-rank active HBM and Grace expert counts.
# --------------------------------------------------------------------------


def assign_off(
    active: np.ndarray, primary: np.ndarray, is_hot: np.ndarray, secondary: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    hbm = np.zeros(EP, dtype=np.int64)
    grace = np.zeros(EP, dtype=np.int64)
    for index in range(active.size):
        if is_hot[index]:
            hbm[primary[index]] += 1
        else:
            grace[primary[index]] += 1
    return hbm, grace, 0


def assign_greedy_v1(
    active: np.ndarray,
    primary: np.ndarray,
    is_hot: np.ndarray,
    secondary: np.ndarray,
    counts: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """The reverted runtime greedy: fixed tasks first, then flexible by key."""
    hbm = np.zeros(EP, dtype=np.int64)
    grace = np.zeros(EP, dtype=np.int64)
    moved = 0

    order = np.lexsort((active, -counts, is_hot.astype(np.int8)))
    fixed = [i for i in order if secondary[i] < 0]
    flexible = [i for i in order if secondary[i] >= 0]

    for index in fixed:
        if is_hot[index]:
            hbm[primary[index]] += 1
        else:
            grace[primary[index]] += 1

    for index in flexible:
        best = None
        for rank in (int(primary[index]), int(secondary[index])):
            use_hbm = rank == primary[index] and is_hot[index]
            if use_hbm:
                hbm[rank] += 1
            else:
                grace[rank] += 1
            times = rank_times(hbm, grace, model)
            candidate = (float(times.max()), float(times.sum()), rank, use_hbm)
            if use_hbm:
                hbm[rank] -= 1
            else:
                grace[rank] -= 1
            if best is None or candidate < best:
                best = candidate
        _, _, rank, use_hbm = best
        if use_hbm:
            hbm[rank] += 1
        else:
            grace[rank] += 1
        moved += rank != primary[index]
    return hbm, grace, moved


def assign_exact_cold(
    active: np.ndarray,
    primary: np.ndarray,
    is_hot: np.ndarray,
    secondary: np.ndarray,
    cross_check: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Hot pinned to primary; cold experts balanced by exact min-max orientation."""
    hbm = np.zeros(EP, dtype=np.int64)
    offsets = np.zeros(EP, dtype=np.int64)
    pair_counts = np.zeros(len(PAIRS), dtype=np.int64)
    edges: list[list[int]] = [[] for _ in PAIRS]

    for index in range(active.size):
        rank = int(primary[index])
        if is_hot[index]:
            hbm[rank] += 1
            continue
        replica = int(secondary[index])
        if replica < 0:
            offsets[rank] += 1
            continue
        pair = (min(rank, replica), max(rank, replica))
        slot = PAIR_INDEX[pair]
        pair_counts[slot] += 1
        edges[slot].append(index)

    grace = offsets.copy()
    moved = 0
    if pair_counts.sum():
        _, split = solve_min_max(offsets, pair_counts, cross_check)
        for slot, (low, high) in enumerate(PAIRS):
            members = edges[slot]
            to_low = int(split[slot])
            # Deterministic realisation: ascending expert id fills the lower
            # rank first. ``edges`` is built in ascending order already.
            for position, index in enumerate(members):
                rank = low if position < to_low else high
                grace[rank] += 1
                moved += rank != primary[index]
    return hbm, grace, moved


def assign_exact_hatch(
    active: np.ndarray,
    primary: np.ndarray,
    is_hot: np.ndarray,
    secondary: np.ndarray,
    model: str,
    cross_check: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """exact_cold, then hot->secondary moves that strictly lower the maximum."""
    hbm, grace, moved = assign_exact_cold(
        active, primary, is_hot, secondary, cross_check
    )
    movable = [
        index
        for index in range(active.size)
        if is_hot[index] and secondary[index] >= 0
    ]
    improved = True
    while improved:
        improved = False
        current = float(rank_times(hbm, grace, model).max())
        for index in movable:
            source = int(primary[index])
            target = int(secondary[index])
            if hbm[source] == 0:
                continue
            hbm[source] -= 1
            grace[target] += 1
            candidate = float(rank_times(hbm, grace, model).max())
            if candidate < current:
                moved += 1
                improved = True
                break
            hbm[source] += 1
            grace[target] -= 1
    return hbm, grace, moved


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def layer_inputs(
    counts: np.ndarray, layer_owners: np.ndarray, layer_hot: np.ndarray,
    layer_secondary: np.ndarray,
) -> tuple[np.ndarray, ...]:
    active = np.flatnonzero(counts)
    return (
        active,
        layer_owners[active].astype(np.int64),
        layer_hot[active],
        layer_secondary[active].astype(np.int64),
        counts[active].astype(np.int64),
    )


def evaluate(
    counts: np.ndarray,
    owners: np.ndarray,
    hot: np.ndarray,
    secondary: np.ndarray,
    policy: str,
    model: str,
    cross_check: bool,
) -> dict:
    steps, layers = counts.shape[0], counts.shape[1]
    span = np.zeros((steps, len(COST_MODELS)))
    skew = np.zeros((steps, len(COST_MODELS)))
    hbm_activations = 0
    grace_activations = 0
    moved_total = 0
    task_total = 0

    for step in range(steps):
        for layer in range(layers):
            active, primary, is_hot, replica, route = layer_inputs(
                counts[step, layer], owners[layer], hot[layer], secondary[layer]
            )
            if active.size == 0:
                continue
            if policy == "off":
                hbm, grace, moved = assign_off(active, primary, is_hot, replica)
            elif policy == "greedy_v1":
                hbm, grace, moved = assign_greedy_v1(
                    active, primary, is_hot, replica, route, model
                )
            elif policy == "exact_cold":
                hbm, grace, moved = assign_exact_cold(
                    active, primary, is_hot, replica, cross_check
                )
            elif policy == "exact_hatch":
                hbm, grace, moved = assign_exact_hatch(
                    active, primary, is_hot, replica, model, cross_check
                )
            else:
                raise ValueError(f"Unknown policy: {policy}")

            for index, scoring in enumerate(COST_MODELS):
                times = rank_times(hbm, grace, scoring)
                span[step, index] += float(times.max())
                skew[step, index] += float(times.max() - times.mean())
            hbm_activations += int(hbm.sum())
            grace_activations += int(grace.sum())
            moved_total += moved
            task_total += int(active.size)

    result = {
        "assignment_cost_model": model,
        "secondary_task_percent": 100 * moved_total / task_total,
        "hbm_task_percent": 100 * hbm_activations / task_total,
        "grace_task_percent": 100 * grace_activations / task_total,
        "grace_activations_per_step": grace_activations / steps,
        "hbm_activations_per_step": hbm_activations / steps,
    }
    for index, scoring in enumerate(COST_MODELS):
        result[f"span_ms_{scoring}"] = float(span[:, index].mean() / 1000)
        result[f"rank_skew_ms_{scoring}"] = float(skew[:, index].mean() / 1000)
    return result


def policy_runs() -> Iterator[tuple[str, str]]:
    """Policy/model pairs. Model-independent policies are evaluated once."""
    yield "off", "n/a"
    yield "exact_cold", "n/a"
    for model in ("legacy", "chain"):
        yield "greedy_v1", model
        yield "exact_hatch", model


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    heldout = load_requests(args.trace_dir, "heldout")
    workloads = {
        "c1": route_counts(sample_steps(heldout, args.heldout_c1_steps, 1, rng)),
        "c4": route_counts(sample_steps(heldout, args.heldout_c4_steps, 4, rng)),
    }
    override = (
        load_profile(args.profile, args.hot_slots_per_rank) if args.profile else None
    )

    results: dict = {
        "trace_dir": str(args.trace_dir),
        "profile": str(args.profile) if args.profile else None,
        "samples": {name: int(c.shape[0]) for name, c in workloads.items()},
        "cost_models": list(COST_MODELS),
        "placements": {},
    }

    for placement_path in args.placement:
        owners, hot, secondary = load_placement(placement_path)
        if override is not None:
            owners, hot = override
        copies = int((secondary >= 0).sum()) // EP
        hot_per_rank = [int((hot & (owners == rank)).sum()) for rank in range(EP)]
        entry: dict = {
            "copies_per_rank": copies,
            "hot_slots_by_rank": hot_per_rank,
            "workloads": {},
        }
        for name, counts in workloads.items():
            arms = {}
            for policy, model in policy_runs():
                key = policy if model == "n/a" else f"{policy}:{model}"
                arms[key] = evaluate(
                    counts,
                    owners,
                    hot,
                    secondary,
                    policy,
                    model,
                    cross_check=True,
                )
                print(
                    f"{placement_path.name} {name} {key:22s} "
                    f"span_chain={arms[key]['span_ms_chain']:8.3f} ms  "
                    f"span_legacy={arms[key]['span_ms_legacy']:8.3f} ms  "
                    f"grace/step={arms[key]['grace_activations_per_step']:8.1f}",
                    flush=True,
                )
            entry["workloads"][name] = arms
        results["placements"][str(placement_path)] = entry

    if args.output:
        args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
