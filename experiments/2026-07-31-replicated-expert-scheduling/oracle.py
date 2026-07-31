#!/usr/bin/env python3
"""Replay routed-expert traces with Grace-resident secondary copies."""

import argparse
import json
from pathlib import Path

import numpy as np

EP = 4
ROUTED_START = 3
VERIFY_TOKENS = 4
HOT_US = 1280.0 / 75
COLD_US = 3467.0 / 75
EXPERT_BYTES = 20_054_024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--hot-slots-per-rank", type=int, default=3176)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--c4-eval-steps", type=int, default=150)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=(985, 1970, 2955, 3300, 3546, 3940),
    )
    parser.add_argument("--oracle-cases", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--placement-dir", type=Path)
    return parser.parse_args()


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
            np.asarray(routes[:, ROUTED_START:, :]).reshape(-1, VERIFY_TOKENS, 75, 8)
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
    counts = np.zeros((steps.shape[0], steps.shape[2], 256), dtype=np.uint8)
    for step in range(steps.shape[0]):
        for layer in range(steps.shape[2]):
            counts[step, layer] = np.bincount(
                steps[step, :, layer, :].reshape(-1), minlength=256
            )
    return counts


def load_profile(path: Path, hot_slots_per_rank: int) -> tuple[np.ndarray, np.ndarray]:
    profile = json.loads(path.read_text())
    if profile["ep_size"] != EP or profile["num_experts"] != 256:
        raise ValueError("The oracle currently requires EP4 with 256 experts")
    owners = np.asarray(profile["owners"], dtype=np.int8)
    hot_lists = [list(experts) for experts in profile["hot_experts"]]
    hot = np.zeros_like(owners, dtype=bool)
    for layer, experts in enumerate(hot_lists):
        hot[layer, np.asarray(experts)] = True
    if owners.shape != (75, 256):
        raise ValueError(f"Unexpected owner shape: {owners.shape}")
    for rank in range(EP):
        current = int(np.count_nonzero(hot & (owners == rank)))
        if current > hot_slots_per_rank:
            remaining = current - hot_slots_per_rank
            while remaining:
                for layer, experts in enumerate(hot_lists):
                    for index in range(len(experts) - 1, -1, -1):
                        expert = experts[index]
                        if owners[layer, expert] == rank:
                            experts.pop(index)
                            hot[layer, expert] = False
                            remaining -= 1
                            break
                    if not remaining:
                        break
        remaining = hot_slots_per_rank - current
        if current > hot_slots_per_rank:
            remaining = 0
        while remaining:
            progressed = False
            for layer in range(75):
                for expert in np.flatnonzero(owners[layer] == rank):
                    if not hot[layer, expert]:
                        hot[layer, expert] = True
                        remaining -= 1
                        progressed = True
                        break
                if not remaining:
                    break
            if not progressed:
                raise ValueError("HBM target exceeds locally owned expert slots")
    return owners, hot


def add_task(
    hbm: np.ndarray,
    grace: np.ndarray,
    rank: int,
    use_hbm: bool,
    sign: int = 1,
) -> None:
    chain = hbm if use_hbm else grace
    chain[rank] += sign * (HOT_US if use_hbm else COLD_US)


def rank_times(hbm: np.ndarray, grace: np.ndarray) -> np.ndarray:
    return np.maximum(hbm, grace)


def assign_layer(
    counts: np.ndarray,
    layer: int,
    owners: np.ndarray,
    hot: np.ndarray,
    secondary: np.ndarray,
    policy: str,
    step: int,
) -> tuple[float, float, int, int, int]:
    active = np.flatnonzero(counts)
    hbm = np.zeros(EP)
    grace = np.zeros(EP)
    token_load = np.zeros(EP, dtype=np.int32)
    secondary_uses = 0
    hbm_uses = 0
    grace_uses = 0

    if policy == "greedy":
        task_cost = np.where(hot[layer, active], HOT_US, COLD_US)
        token_counts = counts[active].astype(np.int16)
        order = active[np.lexsort((active, -token_counts, -task_cost))]
    elif policy == "least_loaded":
        token_counts = counts[active].astype(np.int16)
        order = active[np.lexsort((active, -token_counts))]
    else:
        order = active

    for expert in order:
        primary = int(owners[layer, expert])
        replica = int(secondary[layer, expert])
        candidates = (primary,) if replica < 0 else (primary, replica)

        if policy == "fixed" or replica < 0:
            chosen = primary
        elif policy == "hash":
            value = (
                step * 0x9E3779B1 + layer * 0x85EBCA77 + int(expert) * 0xC2B2AE3D
            ) & 0xFFFFFFFF
            chosen = candidates[value & 1]
        elif policy == "least_loaded":
            chosen = min(candidates, key=lambda rank: (token_load[rank], rank))
        elif policy == "greedy":
            choices = []
            for rank in candidates:
                use_hbm = rank == primary and hot[layer, expert]
                add_task(hbm, grace, rank, use_hbm)
                times = rank_times(hbm, grace)
                choices.append((times.max(), times.sum(), rank))
                add_task(hbm, grace, rank, use_hbm, -1)
            chosen = min(choices)[2]
        else:
            raise ValueError(f"Unknown policy: {policy}")

        use_hbm = chosen == primary and hot[layer, expert]
        add_task(hbm, grace, chosen, use_hbm)
        token_load[chosen] += int(counts[expert])
        secondary_uses += chosen != primary
        hbm_uses += use_hbm
        grace_uses += not use_hbm

    times = rank_times(hbm, grace)
    return (
        float(times.max()),
        float(times.max() - times.mean()),
        secondary_uses,
        hbm_uses,
        grace_uses,
    )


def evaluate(
    counts: np.ndarray,
    owners: np.ndarray,
    hot: np.ndarray,
    secondary: np.ndarray,
    policy: str,
) -> dict:
    totals = np.zeros((counts.shape[0], 2))
    uses = np.zeros(3, dtype=np.int64)
    for step in range(counts.shape[0]):
        for layer in range(counts.shape[1]):
            result = assign_layer(
                counts[step, layer],
                layer,
                owners,
                hot,
                secondary,
                policy,
                step,
            )
            totals[step] += result[:2]
            uses += result[2:]
    tasks = int(uses[1] + uses[2])
    return {
        "span_ms": float(totals[:, 0].mean() / 1000),
        "rank_skew_ms": float(totals[:, 1].mean() / 1000),
        "secondary_task_percent": 100 * int(uses[0]) / tasks,
        "hbm_task_percent": 100 * int(uses[1]) / tasks,
        "grace_task_percent": 100 * int(uses[2]) / tasks,
    }


def score_replicas(
    counts: np.ndarray, owners: np.ndarray, hot: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    span_gain = np.zeros((75, 256, EP), dtype=np.float64)
    skew_gain = np.zeros_like(span_gain)
    activations = np.zeros((75, 256), dtype=np.int32)
    for step in range(counts.shape[0]):
        for layer in range(75):
            active = np.flatnonzero(counts[step, layer])
            activations[layer, active] += 1
            hbm = np.zeros(EP)
            grace = np.zeros(EP)
            for expert in active:
                rank = int(owners[layer, expert])
                add_task(hbm, grace, rank, hot[layer, expert])
            before = rank_times(hbm, grace)
            for expert in active:
                source = int(owners[layer, expert])
                source_hbm = bool(hot[layer, expert])
                for destination in range(EP):
                    if destination == source:
                        continue
                    add_task(hbm, grace, source, source_hbm, -1)
                    add_task(hbm, grace, destination, False)
                    after = rank_times(hbm, grace)
                    span_gain[layer, expert, destination] += max(
                        0.0, before.max() - after.max()
                    )
                    skew_gain[layer, expert, destination] += max(
                        0.0,
                        (before.max() - before.mean()) - (after.max() - after.mean()),
                    )
                    add_task(hbm, grace, destination, False, -1)
                    add_task(hbm, grace, source, source_hbm)
    return span_gain, skew_gain, activations


def place_replicas(
    owners: np.ndarray,
    span_gain: np.ndarray,
    skew_gain: np.ndarray,
    activations: np.ndarray,
    budget: int,
) -> np.ndarray:
    candidates = []
    for layer in range(75):
        for expert in range(256):
            for destination in range(EP):
                if destination != owners[layer, expert]:
                    candidates.append(
                        (
                            -span_gain[layer, expert, destination],
                            -skew_gain[layer, expert, destination],
                            -activations[layer, expert],
                            layer,
                            expert,
                            destination,
                        )
                    )
    candidates.sort()
    secondary = np.full((75, 256), -1, dtype=np.int8)
    used = np.zeros(EP, dtype=np.int32)
    for _, _, _, layer, expert, destination in candidates:
        if secondary[layer, expert] >= 0 or used[destination] >= budget:
            continue
        secondary[layer, expert] = destination
        used[destination] += 1
        if np.all(used == budget):
            break
    if np.any(used != budget):
        raise RuntimeError(f"Could not fill replica budget: {used} != {budget}")
    return secondary


def exact_layer_span(
    counts: np.ndarray,
    layer: int,
    owners: np.ndarray,
    hot: np.ndarray,
    secondary: np.ndarray,
    node_limit: int = 200_000,
) -> tuple[float | None, int]:
    active = np.flatnonzero(counts)
    fixed = [expert for expert in active if secondary[layer, expert] < 0]
    flexible = [expert for expert in active if secondary[layer, expert] >= 0]
    hbm = np.zeros(EP)
    grace = np.zeros(EP)
    for expert in fixed:
        rank = int(owners[layer, expert])
        add_task(hbm, grace, rank, hot[layer, expert])

    flexible.sort(
        key=lambda expert: (
            -(HOT_US if hot[layer, expert] else COLD_US),
            -int(counts[expert]),
            int(expert),
        )
    )
    incumbent = assign_layer(counts, layer, owners, hot, secondary, "greedy", 0)[0]
    nodes = 0
    remaining_min = np.cumsum(
        [
            min(HOT_US if hot[layer, expert] else COLD_US, COLD_US)
            for expert in reversed(flexible)
        ]
    )[::-1]

    def search(index: int) -> None:
        nonlocal incumbent, nodes
        nodes += 1
        if nodes > node_limit:
            return
        current = rank_times(hbm, grace)
        remaining = 0 if index == len(flexible) else remaining_min[index]
        lower_bound = max(
            float(current.max()),
            (hbm.sum() + grace.sum() + remaining) / 8,
        )
        if lower_bound >= incumbent:
            return
        if index == len(flexible):
            incumbent = float(current.max())
            return
        expert = flexible[index]
        primary = int(owners[layer, expert])
        candidates = (primary, int(secondary[layer, expert]))
        ranked = []
        for rank in candidates:
            use_hbm = rank == primary and hot[layer, expert]
            add_task(hbm, grace, rank, use_hbm)
            ranked.append((rank_times(hbm, grace).max(), rank, use_hbm))
            add_task(hbm, grace, rank, use_hbm, -1)
        for _, rank, use_hbm in sorted(ranked):
            add_task(hbm, grace, rank, use_hbm)
            search(index + 1)
            add_task(hbm, grace, rank, use_hbm, -1)
            if nodes > node_limit:
                break

    search(0)
    return (None if nodes > node_limit else incumbent), nodes


def oracle_check(
    counts: np.ndarray,
    owners: np.ndarray,
    hot: np.ndarray,
    secondary: np.ndarray,
    requested: int,
) -> dict:
    gaps = []
    nodes = []
    for step in range(counts.shape[0]):
        for layer in range(75):
            flexible = np.count_nonzero(
                (counts[step, layer] > 0) & (secondary[layer] >= 0)
            )
            if not 1 <= flexible <= 22:
                continue
            greedy = assign_layer(
                counts[step, layer],
                layer,
                owners,
                hot,
                secondary,
                "greedy",
                step,
            )[0]
            exact, visited = exact_layer_span(
                counts[step, layer], layer, owners, hot, secondary
            )
            if exact is None:
                continue
            gaps.append(greedy - exact)
            nodes.append(visited)
            if len(gaps) == requested:
                break
        if len(gaps) == requested:
            break
    return {
        "completed_cases": len(gaps),
        "mean_greedy_gap_us": float(np.mean(gaps)) if gaps else None,
        "max_greedy_gap_us": float(np.max(gaps)) if gaps else None,
        "mean_search_nodes": float(np.mean(nodes)) if nodes else None,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    train_requests = load_requests(args.trace_dir, "train")
    heldout_requests = load_requests(args.trace_dir, "heldout")
    owners, hot = load_profile(args.profile, args.hot_slots_per_rank)
    profile_data = json.loads(args.profile.read_text())

    train = route_counts(sample_steps(train_requests, args.train_steps, 4, rng))
    c1 = route_counts(sample_steps(heldout_requests, args.eval_steps, 1, rng))
    c4 = route_counts(sample_steps(heldout_requests, args.c4_eval_steps, 4, rng))
    no_replicas = np.full((75, 256), -1, dtype=np.int8)
    control = {
        "c1": evaluate(c1, owners, hot, no_replicas, "fixed"),
        "c4": evaluate(c4, owners, hot, no_replicas, "fixed"),
    }

    print(
        f"control: c1 {control['c1']['span_ms']:.3f} ms, "
        f"c4 {control['c4']['span_ms']:.3f} ms routed span"
    )
    span_gain, skew_gain, activations = score_replicas(train, owners, hot)
    results = []
    for budget in args.budgets:
        secondary = place_replicas(owners, span_gain, skew_gain, activations, budget)
        if args.placement_dir:
            args.placement_dir.mkdir(parents=True, exist_ok=True)
            placement = {
                **profile_data,
                "profile_version": 2,
                "hot_experts": [
                    np.flatnonzero(layer_hot).tolist() for layer_hot in hot
                ],
                "secondary_ranks": secondary.tolist(),
                "optimizer": "replicated-makespan-v1",
            }
            placement_path = args.placement_dir / f"replicas-{budget}.json"
            placement_path.write_text(json.dumps(placement) + "\n")
        row = {
            "copies_per_rank": budget,
            "grace_gb_per_rank": budget * EXPERT_BYTES / 1e9,
            "copy_factor": 1 + budget * EP / (75 * 256),
            "placement": (
                str(placement_path) if args.placement_dir is not None else None
            ),
            "workloads": {},
        }
        for name, workload in (("c1", c1), ("c4", c4)):
            row["workloads"][name] = {
                policy: evaluate(workload, owners, hot, secondary, policy)
                for policy in ("hash", "least_loaded", "greedy")
            }
        row["oracle"] = oracle_check(c1, owners, hot, secondary, args.oracle_cases)
        results.append(row)
        greedy_c1 = row["workloads"]["c1"]["greedy"]["span_ms"]
        greedy_c4 = row["workloads"]["c4"]["greedy"]["span_ms"]
        print(
            f"{budget:4d} copies/rank ({row['copy_factor']:.3f}x): "
            f"greedy c1 {greedy_c1:.3f} "
            f"({greedy_c1 - control['c1']['span_ms']:+.3f}), "
            f"c4 {greedy_c4:.3f} "
            f"({greedy_c4 - control['c4']['span_ms']:+.3f}) ms"
        )

    report = {
        "trace_dir": str(args.trace_dir),
        "profile": str(args.profile),
        "hot_slots_per_rank": args.hot_slots_per_rank,
        "cost_model_us": {"hbm": HOT_US, "grace": COLD_US},
        "samples": {
            "train_c4": args.train_steps,
            "heldout_c1": args.eval_steps,
            "heldout_c4": args.c4_eval_steps,
        },
        "control": control,
        "budgets": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
