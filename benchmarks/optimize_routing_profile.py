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
    total = 0
    cold = 0
    request_critical = []
    request_owner_max = []
    for _, routes in requests:
        token_critical = np.zeros(routes.shape[0], dtype=np.int64)
        token_owner_max = np.zeros(routes.shape[0], dtype=np.int64)
        for layer in range(len(ROUTED_LAYERS)):
            ids = routes[:, layer, :]
            selected_owners = owners[layer, ids]
            selected_hot = hot[layer, ids]
            total += ids.size
            cold += int((~selected_hot).sum())
            for token in range(routes.shape[0]):
                owner_counts = np.bincount(selected_owners[token], minlength=EP_SIZE)
                cold_counts = np.bincount(
                    selected_owners[token],
                    weights=(~selected_hot[token]).astype(np.int64),
                    minlength=EP_SIZE,
                )
                token_owner_max[token] += int(owner_counts.max())
                token_critical[token] += int(cold_counts.max())
        request_critical.append(float(token_critical.mean()))
        request_owner_max.append(float(token_owner_max.mean()))
    return {
        "routing_cold_hit_rate": cold / total,
        "mean_token_cold_critical_count": float(np.mean(request_critical)),
        "cvar95_request_cold_critical_count": float(
            np.mean(
                sorted(request_critical)[math.ceil(0.95 * len(request_critical)) - 1 :]
            )
        ),
        "mean_token_owner_max_count": float(np.mean(request_owner_max)),
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
    optimized_hot = optimize_hot(counts, optimized_owners, args.hot_slots_per_rank)
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
        "optimizer": "greedy-balanced-owner+frequency-residency-v1",
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
        "training": {
            "linear_even": evaluate(training, baseline_owners, baseline_hot),
            "optimized": evaluate(training, optimized_owners, optimized_hot),
        },
        "heldout": {
            "linear_even": evaluate(heldout, baseline_owners, baseline_hot),
            "optimized": evaluate(heldout, optimized_owners, optimized_hot),
        },
    }
    args.output_profile.write_text(json.dumps(profile, indent=2) + "\n")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
