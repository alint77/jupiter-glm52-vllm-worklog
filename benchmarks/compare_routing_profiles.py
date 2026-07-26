#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from optimize_routing_profile import ROUTED_LAYERS, evaluate, load_requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="append", required=True)
    return parser.parse_args()


def load_profile(value: str) -> tuple[str, np.ndarray, np.ndarray]:
    label, separator, path = value.partition("=")
    if not separator:
        raise ValueError("Profiles must use LABEL=PATH")
    data = json.loads(Path(path).read_text())
    owners = np.asarray(data["owners"], dtype=np.int16)
    hot = np.zeros_like(owners, dtype=bool)
    for layer, expert_ids in enumerate(data["hot_experts"]):
        hot[layer, expert_ids] = True
    return label, owners, hot


def aggregate(metrics: dict) -> dict:
    return {
        "hot_route_coverage": 1 - metrics["routing_cold_hit_rate"],
        "mean_token_cold_critical_count": metrics[
            "mean_token_cold_critical_count"
        ],
        "cvar95_request_cold_critical_count": metrics[
            "cvar95_request_cold_critical_count"
        ],
        "tail_objective": metrics["tail_objective"],
        "mean_token_owner_max_count": metrics["mean_token_owner_max_count"],
    }


def main() -> None:
    args = parse_args()
    requests = load_requests(args.trace_dir)
    metadata = {
        record["request_hash"]: record
        for record in json.loads((args.trace_dir / "manifest.json").read_text())
    }
    splits = {
        split: [
            request
            for request in requests
            if metadata[request[0]]["split"] == split
        ]
        for split in ("train", "heldout")
    }
    results = {}
    for value in args.profile:
        label, owners, hot = load_profile(value)
        if owners.shape != (len(ROUTED_LAYERS), 256):
            raise ValueError(f"{label} has an invalid owner shape")
        results[label] = {
            split: aggregate(evaluate(split_requests, owners, hot))
            for split, split_requests in splits.items()
        }
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
