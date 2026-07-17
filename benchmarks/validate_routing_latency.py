#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-report", type=Path, required=True)
    parser.add_argument("--linear-replay", type=Path, required=True)
    parser.add_argument("--profiled-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def replay_map(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {request["request_hash"]: request["tpot_ms"] for request in data["requests"]}


def feature_rows(metrics: dict) -> dict[str, list[float]]:
    return {
        request["request_hash"]: [
            1.0,
            request["mean_token_cold_critical_count"],
        ]
        for request in metrics["requests"]
    }


def main() -> None:
    args = parse_args()
    report = json.loads(args.optimizer_report.read_text())
    linear_replay = replay_map(args.linear_replay)
    profiled_replay = replay_map(args.profiled_replay)

    train_linear = feature_rows(report["training"]["linear_even"])
    train_profiled = feature_rows(report["training"]["optimized"])
    train_hashes = list(train_linear)
    x_train = np.array(
        [train_linear[request_hash] for request_hash in train_hashes]
        + [train_profiled[request_hash] for request_hash in train_hashes]
    )
    y_train = np.array(
        [linear_replay[request_hash] for request_hash in train_hashes]
        + [profiled_replay[request_hash] for request_hash in train_hashes]
    )
    coefficients, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)

    heldout_linear = feature_rows(report["heldout"]["linear_even"])
    heldout_profiled = feature_rows(report["heldout"]["optimized"])
    heldout = []
    for request_hash in heldout_linear:
        for placement, features, replay in (
            ("linear_even", heldout_linear[request_hash], linear_replay),
            ("profiled", heldout_profiled[request_hash], profiled_replay),
        ):
            predicted = float(np.dot(coefficients, features))
            observed = replay[request_hash]
            heldout.append(
                {
                    "request_hash": request_hash,
                    "placement": placement,
                    "predicted_tpot_ms": predicted,
                    "observed_tpot_ms": observed,
                    "relative_error": abs(predicted - observed) / observed,
                }
            )
    train_prediction = x_train @ coefficients
    result = {
        "model": "affine(cold_critical_count)",
        "coefficients": coefficients.tolist(),
        "training_mean_relative_error": float(
            np.mean(np.abs(train_prediction - y_train) / y_train)
        ),
        "heldout": heldout,
        "heldout_max_relative_error": max(row["relative_error"] for row in heldout),
    }
    result["within_20_percent"] = result["heldout_max_relative_error"] <= 0.20
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
