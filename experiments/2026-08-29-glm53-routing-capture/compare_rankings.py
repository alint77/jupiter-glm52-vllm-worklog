#!/usr/bin/env python3
"""Score placement profiles against the same held-out GLM-5.3 routing traces.

Answers the question the re-derivation exists to answer: how much does the
GLM-5.2 ranking that every 5.3 profile currently ships cost on real 5.3 agentic
coding traffic? Every profile is evaluated on the held-out split only, with the
metrics `optimize_routing_profile.py` optimises.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from agent_space.benchmarks.optimize_routing_profile import (  # noqa: E402
    NUM_EXPERTS,
    ROUTED_LAYERS,
    evaluate,
    load_requests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable; scores PATH under LABEL",
    )
    parser.add_argument("--split", choices=("heldout", "train", "all"), default="heldout")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def profile_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    if data["routed_layers"] != list(ROUTED_LAYERS):
        raise ValueError(f"{path}: routed layers do not match")
    owners = np.asarray(data["owners"], dtype=np.int16)
    hot = np.zeros((len(ROUTED_LAYERS), NUM_EXPERTS), dtype=bool)
    for layer, experts in enumerate(data["hot_experts"]):
        hot[layer, experts] = True
    return owners, hot


def main() -> None:
    args = parse_args()
    manifest = {
        record["request_hash"]: record
        for record in json.loads((args.trace_dir / "manifest.json").read_text())
    }
    requests = load_requests(args.trace_dir)
    if args.split != "all":
        requests = [
            request
            for request in requests
            if manifest[request[0]]["split"] == args.split
        ]
    if not requests:
        raise ValueError(f"No {args.split} requests in {args.trace_dir}")

    results = {}
    hot_sets = {}
    for spec in args.profile:
        label, _, path = spec.partition("=")
        owners, hot = profile_arrays(Path(path))
        results[label] = evaluate(requests, owners, hot)
        hot_sets[label] = hot

    labels = list(results)
    overlap = {}
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            shared = int((hot_sets[left] & hot_sets[right]).sum())
            total = int(hot_sets[left].sum())
            overlap[f"{left} vs {right}"] = {
                "shared_hot_experts": shared,
                "hot_expert_overlap": shared / total if total else 0.0,
            }

    report = {
        "split": args.split,
        "requests": len(requests),
        "routed_positions": sum(request[1].shape[0] for request in requests),
        "profiles": results,
        "hot_set_overlap": overlap,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
