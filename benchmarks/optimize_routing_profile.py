#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vllm.model_executor.model_loader.tiered_moe_manifest import (
    build_glm_w4a16_manifest,
)

ROUTED_LAYERS = tuple(range(3, 78))
NUM_EXPERTS = 256
EP_SIZE = 4
EXPERTS_PER_RANK = NUM_EXPERTS // EP_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--hot-slots-per-rank", type=int, required=True)
    parser.add_argument("--train-requests", type=int)
    return parser.parse_args()


def load_requests(trace_dir: Path) -> list[tuple[str, np.ndarray]]:
    requests = []
    for path in sorted(trace_dir.glob("request-*.npz")):
        with np.load(path) as data:
            routes = data["routes"][:, ROUTED_LAYERS, :].astype(np.int64)
            request_hash = str(data["request_hash"])
        if routes.ndim != 3 or routes.shape[1:] != (len(ROUTED_LAYERS), 8):
            raise ValueError(f"Invalid trace shape in {path}: {routes.shape}")
        requests.append((request_hash, routes))
    if len(requests) < 2:
        raise ValueError("At least two trace requests are required")
    return requests


def route_counts(requests: list[tuple[str, np.ndarray]]) -> np.ndarray:
    counts = np.zeros((len(ROUTED_LAYERS), NUM_EXPERTS), dtype=np.int64)
    for _, routes in requests:
        for layer in range(len(ROUTED_LAYERS)):
            counts[layer] += np.bincount(
                routes[:, layer, :].reshape(-1), minlength=NUM_EXPERTS
            )
    return counts


def optimize_owners(counts: np.ndarray) -> np.ndarray:
    owners = np.empty_like(counts, dtype=np.int16)
    for layer in range(len(ROUTED_LAYERS)):
        rank_load = np.zeros(EP_SIZE, dtype=np.int64)
        rank_size = np.zeros(EP_SIZE, dtype=np.int64)
        expert_order = sorted(
            range(NUM_EXPERTS), key=lambda expert: (-counts[layer, expert], expert)
        )
        for expert in expert_order:
            rank = min(
                (rank for rank in range(EP_SIZE) if rank_size[rank] < EXPERTS_PER_RANK),
                key=lambda candidate: (rank_load[candidate], candidate),
            )
            owners[layer, expert] = rank
            rank_load[rank] += counts[layer, expert]
            rank_size[rank] += 1
    return owners


def linear_owners() -> np.ndarray:
    row = np.repeat(np.arange(EP_SIZE, dtype=np.int16), EXPERTS_PER_RANK)
    return np.tile(row, (len(ROUTED_LAYERS), 1))


def optimize_hot(
    counts: np.ndarray, owners: np.ndarray, hot_slots_per_rank: int
) -> np.ndarray:
    total_slots = len(ROUTED_LAYERS) * EXPERTS_PER_RANK
    if not 0 <= hot_slots_per_rank <= total_slots:
        raise ValueError("Hot slot budget is outside the per-rank capacity")
    hot = np.zeros_like(owners, dtype=bool)
    for rank in range(EP_SIZE):
        candidates = [
            (int(counts[layer, expert]), layer, expert)
            for layer in range(len(ROUTED_LAYERS))
            for expert in range(NUM_EXPERTS)
            if owners[layer, expert] == rank
        ]
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        for _, layer, expert in candidates[:hot_slots_per_rank]:
            hot[layer, expert] = True
    return hot


def _route_state(
    requests: list[tuple[str, np.ndarray]], owners: np.ndarray, hot: np.ndarray
) -> tuple[np.ndarray, list[slice], np.ndarray, np.ndarray]:
    routes = np.concatenate([request[1] for request in requests])
    request_slices = []
    offset = 0
    for _, request_routes in requests:
        request_slices.append(slice(offset, offset + request_routes.shape[0]))
        offset += request_routes.shape[0]

    cold_counts = np.zeros(
        (routes.shape[0], len(ROUTED_LAYERS), EP_SIZE), dtype=np.int16
    )
    owner_counts = np.zeros_like(cold_counts)
    for layer in range(len(ROUTED_LAYERS)):
        ids = routes[:, layer, :]
        selected_owners = owners[layer, ids]
        selected_hot = hot[layer, ids]
        for rank in range(EP_SIZE):
            owned = selected_owners == rank
            owner_counts[:, layer, rank] = owned.sum(axis=1)
            cold_counts[:, layer, rank] = (owned & ~selected_hot).sum(axis=1)
    return routes, request_slices, cold_counts, owner_counts


def _request_means(values: np.ndarray, request_slices: list[slice]) -> np.ndarray:
    return np.array([values[request_slice].mean() for request_slice in request_slices])


def _tail_objective(request_means: np.ndarray) -> float:
    tail_start = math.ceil(0.95 * len(request_means)) - 1
    cvar95 = np.sort(request_means)[tail_start:].mean()
    return float(request_means.mean() + 0.25 * cvar95)


def _occurrences(
    routes: np.ndarray,
    cache: dict[tuple[int, int], np.ndarray],
    layer: int,
    expert: int,
) -> np.ndarray:
    key = (layer, expert)
    if key not in cache:
        cache[key] = (routes[:, layer, :] == expert).sum(axis=1)
    return cache[key]


def optimize_hot_tail_aware(
    requests: list[tuple[str, np.ndarray]],
    counts: np.ndarray,
    owners: np.ndarray,
    hot_slots_per_rank: int,
    *,
    candidate_limit: int = 16,
    swap_limit: int = 8,
) -> tuple[np.ndarray, list[dict]]:
    """Improve frequency residency with bounded exact-objective swaps."""
    hot = optimize_hot(counts, owners, hot_slots_per_rank)
    history = []
    for _ in range(swap_limit):
        routes, request_slices, cold_counts, _ = _route_state(requests, owners, hot)
        layer_max = cold_counts.max(axis=2)
        token_time = layer_max.sum(axis=1)
        current_objective = _tail_objective(_request_means(token_time, request_slices))
        other_max = np.empty_like(cold_counts)
        for rank in range(EP_SIZE):
            other_ranks = [
                candidate for candidate in range(EP_SIZE) if candidate != rank
            ]
            other_max[:, :, rank] = cold_counts[:, :, other_ranks].max(axis=2)

        best = None
        occurrence_cache = {}

        for rank in range(EP_SIZE):
            hot_candidates = sorted(
                (
                    (int(counts[layer, expert]), layer, expert)
                    for layer in range(len(ROUTED_LAYERS))
                    for expert in range(NUM_EXPERTS)
                    if owners[layer, expert] == rank and hot[layer, expert]
                ),
                key=lambda item: (item[0], item[1], item[2]),
            )[:candidate_limit]
            cold_candidates = sorted(
                (
                    (int(counts[layer, expert]), layer, expert)
                    for layer in range(len(ROUTED_LAYERS))
                    for expert in range(NUM_EXPERTS)
                    if owners[layer, expert] == rank and not hot[layer, expert]
                ),
                key=lambda item: (-item[0], item[1], item[2]),
            )[:candidate_limit]
            for _, hot_layer, hot_expert in hot_candidates:
                hot_occurrences = _occurrences(
                    routes, occurrence_cache, hot_layer, hot_expert
                )
                for _, cold_layer, cold_expert in cold_candidates:
                    cold_occurrences = _occurrences(
                        routes, occurrence_cache, cold_layer, cold_expert
                    )
                    if hot_layer == cold_layer:
                        new_rank = (
                            cold_counts[:, hot_layer, rank]
                            + hot_occurrences
                            - cold_occurrences
                        )
                        delta = (
                            np.maximum(other_max[:, hot_layer, rank], new_rank)
                            - layer_max[:, hot_layer]
                        )
                    else:
                        hot_rank = cold_counts[:, hot_layer, rank] + hot_occurrences
                        cold_rank = cold_counts[:, cold_layer, rank] - cold_occurrences
                        delta = (
                            np.maximum(other_max[:, hot_layer, rank], hot_rank)
                            - layer_max[:, hot_layer]
                            + np.maximum(other_max[:, cold_layer, rank], cold_rank)
                            - layer_max[:, cold_layer]
                        )
                    objective = _tail_objective(
                        _request_means(token_time + delta, request_slices)
                    )
                    candidate = (
                        objective,
                        rank,
                        hot_layer,
                        hot_expert,
                        cold_layer,
                        cold_expert,
                    )
                    if best is None or candidate < best:
                        best = candidate
        if best is None or best[0] >= current_objective - 1e-12:
            break
        objective, rank, hot_layer, hot_expert, cold_layer, cold_expert = best
        hot[hot_layer, hot_expert] = False
        hot[cold_layer, cold_expert] = True
        history.append(
            {
                "rank": rank,
                "demoted": [ROUTED_LAYERS[hot_layer], hot_expert],
                "promoted": [ROUTED_LAYERS[cold_layer], cold_expert],
                "objective_before": current_objective,
                "objective_after": objective,
            }
        )
    return hot, history


def even_hot(owners: np.ndarray, hot_slots_per_rank: int) -> np.ndarray:
    hot = np.zeros_like(owners, dtype=bool)
    base, remainder = divmod(hot_slots_per_rank, len(ROUTED_LAYERS))
    for layer in range(len(ROUTED_LAYERS)):
        count = base + (layer < remainder)
        for rank in range(EP_SIZE):
            owned = np.flatnonzero(owners[layer] == rank)
            start = layer % len(owned)
            rotated = np.concatenate((owned[start:], owned[:start]))
            hot[layer, rotated[:count]] = True
    return hot


def evaluate(
    requests: list[tuple[str, np.ndarray]], owners: np.ndarray, hot: np.ndarray
) -> dict:
    routes, request_slices, cold_counts, owner_counts = _route_state(
        requests, owners, hot
    )
    token_critical = cold_counts.max(axis=2).sum(axis=1)
    token_owner_max = owner_counts.max(axis=2).sum(axis=1)
    request_critical = _request_means(token_critical, request_slices)
    request_owner_max = _request_means(token_owner_max, request_slices)
    cvar_start = math.ceil(0.95 * len(request_critical)) - 1
    return {
        "routing_cold_hit_rate": float(
            (~hot[np.arange(len(ROUTED_LAYERS))[None, :, None], routes]).mean()
        ),
        "mean_token_cold_critical_count": float(np.mean(request_critical)),
        "cvar95_request_cold_critical_count": float(
            np.sort(request_critical)[cvar_start:].mean()
        ),
        "tail_objective": _tail_objective(request_critical),
        "mean_token_owner_max_count": float(np.mean(request_owner_max)),
        "requests": [
            {
                "request_hash": request_hash,
                "mean_token_cold_critical_count": float(request_critical[index]),
                "mean_token_owner_max_count": float(request_owner_max[index]),
            }
            for index, (request_hash, _) in enumerate(requests)
        ],
    }


def main() -> None:
    args = parse_args()
    requests = load_requests(args.trace_dir)
    train_count = args.train_requests or max(1, len(requests) * 3 // 4)
    if not 0 < train_count < len(requests):
        raise ValueError("Training request count must leave held-out requests")
    training = requests[:train_count]
    heldout = requests[train_count:]
    counts = route_counts(training)

    baseline_owners = linear_owners()
    baseline_hot = even_hot(baseline_owners, args.hot_slots_per_rank)
    optimized_owners = optimize_owners(counts)
    frequency_hot = optimize_hot(counts, optimized_owners, args.hot_slots_per_rank)
    optimized_hot, tail_swaps = optimize_hot_tail_aware(
        training, counts, optimized_owners, args.hot_slots_per_rank
    )
    manifest = build_glm_w4a16_manifest(args.model)
    profile = {
        "profile_version": 1,
        "config_sha256": manifest.config_sha256,
        "index_sha256": manifest.index_sha256,
        "ep_size": EP_SIZE,
        "num_experts": NUM_EXPERTS,
        "routed_layers": list(ROUTED_LAYERS),
        "owners": optimized_owners.tolist(),
        "hot_experts": [
            np.flatnonzero(optimized_hot[layer]).tolist()
            for layer in range(len(ROUTED_LAYERS))
        ],
        "optimizer": "greedy-balanced-owner+bounded-tail-swap-residency-v2",
        "training_request_hashes": [request[0] for request in training],
        "heldout_request_hashes": [request[0] for request in heldout],
    }
    report = {
        "trace_request_count": len(requests),
        "training_request_count": len(training),
        "heldout_request_count": len(heldout),
        "trace_tokens": sum(request[1].shape[0] for request in requests),
        "hot_slots_per_rank": args.hot_slots_per_rank,
        "capacity_cold_rate": 1
        - args.hot_slots_per_rank / (len(ROUTED_LAYERS) * EXPERTS_PER_RANK),
        "tail_swaps": tail_swaps,
        "training": {
            "linear_even": evaluate(training, baseline_owners, baseline_hot),
            "frequency": evaluate(training, optimized_owners, frequency_hot),
            "optimized": evaluate(training, optimized_owners, optimized_hot),
        },
        "heldout": {
            "linear_even": evaluate(heldout, baseline_owners, baseline_hot),
            "frequency": evaluate(heldout, optimized_owners, frequency_hot),
            "optimized": evaluate(heldout, optimized_owners, optimized_hot),
        },
    }
    args.output_profile.write_text(json.dumps(profile, indent=2) + "\n")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
