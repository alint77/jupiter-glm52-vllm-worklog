#!/usr/bin/env python3
"""Compare distinct activated experts under MTP verification vs independent sequences.

A no-MTP run at concurrency C presents the target model with M=C independent
tokens. An MTP3 run at concurrency C presents M=4C tokens, but as C groups of
four *consecutive* positions from one sequence. Adjacent tokens route to
overlapping experts, so the two regimes do not activate the same number of
distinct experts even at equal M. This script measures the gap on the captured
Claude Code routes, so a no-MTP control can be read with the right correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VERIFY_TOKENS = 4
ROUTED_START = 3
NUM_LAYERS = 75
TOP_K = 8


def load_tokens(trace_dir: Path, split: str) -> np.ndarray:
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    steps = []
    for record in manifest:
        if record["split"] != split:
            continue
        routes = np.load(trace_dir / record["file"], mmap_mode="r")
        if routes.ndim != 3 or routes.shape[1:] != (ROUTED_START + NUM_LAYERS, TOP_K):
            raise ValueError(f"Unexpected route shape {routes.shape} in {record['file']}")
        usable = (routes.shape[0] // VERIFY_TOKENS) * VERIFY_TOKENS
        if not usable:
            continue
        steps.append(
            np.asarray(routes[:usable, ROUTED_START:, :]).reshape(
                -1, VERIFY_TOKENS, NUM_LAYERS, TOP_K
            )
        )
    if not steps:
        raise ValueError(f"No {split} requests in {trace_dir}")
    return np.concatenate(steps)


def mean_distinct(sample: np.ndarray) -> tuple[float, float]:
    """Mean distinct experts per layer over a batch of steps."""
    per_step = []
    for step in sample:
        by_layer = step.reshape(step.shape[0], NUM_LAYERS, TOP_K)
        by_layer = by_layer.transpose(1, 0, 2).reshape(NUM_LAYERS, -1)
        per_step.append(np.mean([len(np.unique(row)) for row in by_layer]))
    return float(np.mean(per_step)), float(np.std(per_step))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(
            "/e/scratch/profound/naeimitabiei1/claude-routing-profile-1047954-108"
        ),
    )
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    steps = load_tokens(args.trace_dir, args.split)
    tokens = steps.reshape(-1, NUM_LAYERS, TOP_K)
    print(f"{args.split}: {len(steps)} MTP-style steps, {len(tokens)} tokens")

    def mtp(groups: int) -> np.ndarray:
        return np.concatenate(
            [steps[rng.integers(len(steps), size=args.samples)] for _ in range(groups)],
            axis=1,
        )

    def independent(width: int) -> np.ndarray:
        return tokens[rng.integers(len(tokens), size=(args.samples, width))]

    cases = (
        ("MTP3 c1 (4 consecutive, 1 seq)", 4, lambda: mtp(1)),
        ("no-MTP c4 (4 independent tokens)", 4, lambda: independent(4)),
        ("MTP3 c4 (4 seq x 4 consecutive)", 16, lambda: mtp(4)),
        ("no-MTP c16 (16 independent tokens)", 16, lambda: independent(16)),
    )
    results = {}
    for label, m, build in cases:
        mean, stdev = mean_distinct(build())
        results[label] = {
            "m": m,
            "distinct_experts_per_layer": mean,
            "stdev": stdev,
            "per_rank": mean / args.ep_size,
        }
        print(
            f"M={m:2d}  {label:36s} distinct/layer {mean:6.2f} +- {stdev:4.2f}"
            f"   per rank {mean / args.ep_size:5.2f}"
        )

    inflation = {}
    for m in (4, 16):
        pair = [row for row in results.values() if row["m"] == m]
        percent = 100 * (
            pair[1]["distinct_experts_per_layer"] / pair[0]["distinct_experts_per_layer"]
            - 1
        )
        print(f"M={m:2d}: independent-sequence inflation {percent:+.1f}%")
        inflation[f"m{m}_percent"] = percent

    if args.json:
        args.json.write_text(
            json.dumps({"regimes": results, "inflation": inflation}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
