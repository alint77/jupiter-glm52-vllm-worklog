#!/usr/bin/env python3
"""Replay the deployed replica profile with the selected shape-dependent costs."""

import argparse
import json
from pathlib import Path

import numpy as np
from oracle import (
    evaluate,
    load_profile,
    load_requests,
    route_counts,
    sample_steps,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    train = load_requests(args.trace_dir, "train")
    heldout = load_requests(args.trace_dir, "heldout")
    sample_steps(train, 300, 4, rng)
    c1 = route_counts(sample_steps(heldout, 300, 1, rng))
    c4 = route_counts(sample_steps(heldout, 150, 4, rng))
    owners, hot = load_profile(args.profile, 2614)
    profile = json.loads(args.profile.read_text())
    secondary = np.asarray(profile["secondary_ranks"], dtype=np.int8)
    no_replicas = np.full((75, 256), -1, dtype=np.int8)

    result = {
        "trace_dir": str(args.trace_dir),
        "profile": str(args.profile),
        "cost_model": "legacy-m4-chain-m16",
        "workloads": {},
    }
    for name, workload in (("c1", c1), ("c4", c4)):
        result["workloads"][name] = {
            "off": evaluate(workload, owners, hot, no_replicas, "fixed"),
            "runtime_greedy": evaluate(
                workload,
                owners,
                hot,
                secondary,
                "runtime_greedy",
            ),
        }

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["workloads"], indent=2))


if __name__ == "__main__":
    main()
