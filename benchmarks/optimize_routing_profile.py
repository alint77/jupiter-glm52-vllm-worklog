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
    parser.add_argument("--mixed-layer-penalty", type=float)
    return parser.parse_args()


def load_requests(trace_dir: Path) -> list[tuple[str, np.ndarray]]:
    requests = []
    for path in sorted(trace_dir.glob("request-*.npz")):
        with np.load(path) as data:
            raw_routes = data["routes"]
            request_hash = str(data["request_hash"])
        if raw_routes.ndim == 3:
            routes = raw_routes[:, ROUTED_LAYERS, :]
        elif raw_routes.ndim == 4:
            routes = raw_routes[:, :, ROUTED_LAYERS, :].transpose(0, 2, 1, 3)
            routes = routes.reshape(routes.shape[0], len(ROUTED_LAYERS), -1)
        else:
            raise ValueError(f"Invalid trace shape in {path}: {raw_routes.shape}")
        routes = routes.astype(np.int64)
        if routes.ndim != 3 or routes.shape[1] != len(ROUTED_LAYERS):
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


def _mixed_layer_count(owners: np.ndarray, hot: np.ndarray) -> int:
    count = 0
    for layer in range(len(ROUTED_LAYERS)):
        rank_hot = [
            int(hot[layer, owners[layer] == rank].sum()) for rank in range(EP_SIZE)
        ]
        count += not (
            all(value == 0 for value in rank_hot)
            or all(value == EXPERTS_PER_RANK for value in rank_hot)
        )
    return count


def _hot_with_full_layers(
    counts: np.ndarray,
    owners: np.ndarray,
    hot_slots_per_rank: int,
    full_layers: set[int],
    candidate_layers: set[int] | None = None,
) -> np.ndarray:
    reserved = len(full_layers) * EXPERTS_PER_RANK
    if reserved > hot_slots_per_rank:
        raise ValueError("Fully hot layers exceed the HBM slot budget")
    hot = np.zeros_like(owners, dtype=bool)
    for layer in full_layers:
        hot[layer] = True
    remaining = hot_slots_per_rank - reserved
    for rank in range(EP_SIZE):
        candidates = [
            (int(counts[layer, expert]), layer, expert)
            for layer in range(len(ROUTED_LAYERS))
            if layer not in full_layers
            and (candidate_layers is None or layer in candidate_layers)
            for expert in range(NUM_EXPERTS)
            if owners[layer, expert] == rank
        ]
        if len(candidates) < remaining:
            raise ValueError("Candidate layers cannot satisfy the HBM slot budget")
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        for _, layer, expert in candidates[:remaining]:
            hot[layer, expert] = True
    return hot


def optimize_hot_layer_concentrated(
    requests: list[tuple[str, np.ndarray]],
    counts: np.ndarray,
    owners: np.ndarray,
    hot_slots_per_rank: int,
    mixed_layer_penalty: float,
) -> tuple[np.ndarray, dict]:
    """Concentrate residency when the saved second-tier launch is worthwhile."""
    if mixed_layer_penalty < 0:
        raise ValueError("Mixed-layer penalty must be non-negative")
    frequency_hot = optimize_hot(counts, owners, hot_slots_per_rank)
    _, _, cold_counts, _ = _route_state(requests, owners, frequency_hot)
    layer_order = sorted(
        range(len(ROUTED_LAYERS)),
        key=lambda layer: (-float(cold_counts[:, layer].max(axis=1).mean()), layer),
    )
    frequency_metrics = evaluate(requests, owners, frequency_hot)
    frequency_mixed = _mixed_layer_count(owners, frequency_hot)
    frequency_candidate = {
        "mode": "per-expert",
        "full_layer_count": 0,
        "full_layers": [],
        "mixed_layer_count": frequency_mixed,
        "cold_layer_count": 0,
        "route_objective": frequency_metrics["tail_objective"],
        "objective": frequency_metrics["tail_objective"]
        + mixed_layer_penalty * frequency_mixed,
    }
    candidates = [frequency_candidate]
    best = (
        (frequency_candidate["objective"], frequency_mixed, 0),
        frequency_hot,
        frequency_candidate,
    )
    max_full_layers = hot_slots_per_rank // EXPERTS_PER_RANK
    for full_count in range(max_full_layers + 1):
        full_layers = set(layer_order[:full_count])
        remaining = hot_slots_per_rank - full_count * EXPERTS_PER_RANK
        partial_count = math.ceil(remaining / EXPERTS_PER_RANK)
        partial_layers = set(layer_order[full_count : full_count + partial_count])
        hot = _hot_with_full_layers(
            counts,
            owners,
            hot_slots_per_rank,
            full_layers,
            partial_layers,
        )
        metrics = evaluate(requests, owners, hot)
        mixed_layers = _mixed_layer_count(owners, hot)
        actual_full_layers = [
            layer for layer in range(len(ROUTED_LAYERS)) if hot[layer].all()
        ]
        cold_layers = [
            layer for layer in range(len(ROUTED_LAYERS)) if not hot[layer].any()
        ]
        objective = metrics["tail_objective"] + mixed_layer_penalty * mixed_layers
        candidate = {
            "mode": "layer-concentrated",
            "full_layer_count": len(actual_full_layers),
            "full_layers": [ROUTED_LAYERS[layer] for layer in actual_full_layers],
            "mixed_layer_count": mixed_layers,
            "cold_layer_count": len(cold_layers),
            "route_objective": metrics["tail_objective"],
            "objective": objective,
        }
        candidates.append(candidate)
        key = (objective, mixed_layers, -full_count)
        if best is None or key < best[0]:
            best = (key, hot, candidate)
    assert best is not None
    return best[1], {
        "mixed_layer_penalty": mixed_layer_penalty,
        "selected": best[2],
        "candidates": candidates,
    }


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


def evaluate_with_domains(
    requests: list[tuple[str, np.ndarray]],
    owners: np.ndarray,
    hot: np.ndarray,
    metadata: dict[str, dict],
) -> dict:
    result = evaluate(requests, owners, hot)
    if not metadata or any(
        request_hash not in metadata or "domain" not in metadata[request_hash]
        for request_hash, _ in requests
    ):
        return result
    domains = sorted({metadata[request_hash]["domain"] for request_hash, _ in requests})
    result["domains"] = {
        domain: evaluate(
            [
                request
                for request in requests
                if metadata[request[0]]["domain"] == domain
            ],
            owners,
            hot,
        )
        for domain in domains
    }
    return result


def main() -> None:
    args = parse_args()
    requests = load_requests(args.trace_dir)
    manifest_path = args.trace_dir / "manifest.json"
    metadata = {}
    if manifest_path.exists():
        metadata = {
            record["request_hash"]: record
            for record in json.loads(manifest_path.read_text())
        }
    has_split = metadata and all("split" in record for record in metadata.values())
    if has_split:
        training = [
            request for request in requests if metadata[request[0]]["split"] == "train"
        ]
        heldout = [
            request
            for request in requests
            if metadata[request[0]]["split"] == "heldout"
        ]
        if args.train_requests is not None and len(training) != args.train_requests:
            raise ValueError("Manifest training split does not match --train-requests")
    else:
        train_count = args.train_requests or max(1, len(requests) * 3 // 4)
        if not 0 < train_count < len(requests):
            raise ValueError("Training request count must leave held-out requests")
        training = requests[:train_count]
        heldout = requests[train_count:]
    if not training or not heldout:
        raise ValueError("Training split must leave held-out requests")
    counts = route_counts(training)

    baseline_owners = linear_owners()
    baseline_hot = even_hot(baseline_owners, args.hot_slots_per_rank)
    optimized_owners = optimize_owners(counts)
    frequency_hot = optimize_hot(counts, optimized_owners, args.hot_slots_per_rank)
    layer_concentration = None
    if args.mixed_layer_penalty is None:
        optimized_hot, tail_swaps = optimize_hot_tail_aware(
            training, counts, optimized_owners, args.hot_slots_per_rank
        )
        optimizer_name = "greedy-balanced-owner+bounded-tail-swap-residency-v2"
    else:
        optimized_hot, layer_concentration = optimize_hot_layer_concentrated(
            training,
            counts,
            optimized_owners,
            args.hot_slots_per_rank,
            args.mixed_layer_penalty,
        )
        tail_swaps = []
        optimizer_name = "greedy-balanced-owner+layer-concentrated-residency-v1"
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
        "optimizer": optimizer_name,
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
        "layer_concentration": layer_concentration,
        "training": {
            "linear_even": evaluate_with_domains(
                training, baseline_owners, baseline_hot, metadata
            ),
            "frequency": evaluate_with_domains(
                training, optimized_owners, frequency_hot, metadata
            ),
            "optimized": evaluate_with_domains(
                training, optimized_owners, optimized_hot, metadata
            ),
        },
        "heldout": {
            "linear_even": evaluate_with_domains(
                heldout, baseline_owners, baseline_hot, metadata
            ),
            "frequency": evaluate_with_domains(
                heldout, optimized_owners, frequency_hot, metadata
            ),
            "optimized": evaluate_with_domains(
                heldout, optimized_owners, optimized_hot, metadata
            ),
        },
    }
    args.output_profile.write_text(json.dumps(profile, indent=2) + "\n")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
