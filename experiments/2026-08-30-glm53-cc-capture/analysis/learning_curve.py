#!/usr/bin/env python3
"""How much capture is enough? Rank on N train requests, score on the holdout.

The question the capture has to answer before it can be stopped: does the
held-out cold-hit rate still improve with more traces, or has it converged.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from agent_space.benchmarks.optimize_routing_profile import (
    NUM_EXPERTS,
    ROUTED_LAYERS,
    evaluate,
    load_requests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.profile.read_text())
    owners = np.asarray(data["owners"], dtype=np.int16)
    hot = np.zeros((len(ROUTED_LAYERS), NUM_EXPERTS), dtype=bool)
    for layer, experts in enumerate(data["hot_experts"]):
        hot[layer, np.asarray(experts, dtype=np.int64)] = True
    per_layer = hot.sum(axis=1)

    manifest = json.loads((args.trace_dir / "manifest.json").read_text())
    split = {r["request_hash"]: r.get("split", "train") for r in manifest}
    requests = load_requests(args.trace_dir)
    train = [r for r in requests if split.get(r[0]) != "heldout"]
    heldout = [r for r in requests if split.get(r[0]) == "heldout"]

    def counts_of(subset):
        c = np.zeros((len(ROUTED_LAYERS), NUM_EXPERTS), dtype=np.int64)
        for _, routes in subset:
            for layer in range(len(ROUTED_LAYERS)):
                c[layer] += np.bincount(
                    routes[:, layer, :].ravel(), minlength=NUM_EXPERTS
                )
        return c

    def rank_hot(c):
        out = np.zeros_like(hot)
        for layer in range(len(ROUTED_LAYERS)):
            out[layer, np.argsort(-c[layer])[: per_layer[layer]]] = True
        return out

    full = rank_hot(counts_of(train))
    rows = []
    sizes = [s for s in (15, 25, 40, 55, 70, 85, len(train)) if s <= len(train)]
    for n in sizes:
        scores, overlaps, positions = [], [], []
        for seed in range(args.seeds if n < len(train) else 1):
            rng = np.random.default_rng(1000 + seed)
            idx = rng.choice(len(train), size=n, replace=False)
            subset = [train[i] for i in idx]
            picked = rank_hot(counts_of(subset))
            scores.append(evaluate(heldout, owners, picked)["routing_cold_hit_rate"])
            overlaps.append(float((picked & full).sum() / int(full.sum())))
            positions.append(sum(r.shape[0] for _, r in subset))
        rows.append(
            {
                "train_requests": n,
                "mean_routed_positions": float(np.mean(positions)),
                "heldout_cold_hit_mean": float(np.mean(scores)),
                "heldout_cold_hit_stdev": float(np.std(scores, ddof=1))
                if len(scores) > 1
                else 0.0,
                "hot_set_overlap_with_full": float(np.mean(overlaps)),
            }
        )
    shipped = evaluate(heldout, owners, hot)["routing_cold_hit_rate"]
    out = {"shipped_profile_heldout": shipped, "curve": rows}
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
