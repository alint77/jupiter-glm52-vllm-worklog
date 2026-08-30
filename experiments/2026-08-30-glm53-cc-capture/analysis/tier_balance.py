#!/usr/bin/env python3
"""Score placements under a max(hot, cold) step model instead of cold-only.

The planner's objective is the cold critical path: per token, per layer, the
max over EP ranks of that rank's cold expert count, summed over layers. Hot
never enters it, so a hot slot in a layer that is already hot-bound is valued
exactly as highly as one in a layer that is cold-bound.

The tiers overlap (`_TIER_BLOCKS_PER_SM = {"hot": 2, "cold": 1}`), so the real
per-layer cost is closer to max(t_hot * hot, t_cold * cold). This scores both
models on the same traces and reports where each layer sits relative to the
balance point t_hot * H == t_cold * C.

Two further corrections the same framing implies:
  * cost is per distinct expert per *step*, not per token: an expert activated
    by several tokens in one step is staged and executed once;
  * a step batches whole requests, so c1 and c4 have different unions.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from agent_space.benchmarks.optimize_routing_profile import (
    EP_SIZE,
    NUM_EXPERTS,
    ROUTED_LAYERS,
    load_requests,
)

# Phase 32 / 2026-08-01 tier-balance measurements, per expert per layer per rank.
T_HOT_US = 9.75
T_COLD_US = 45.32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--tokens-per-step", type=int, default=4, help="MTP3 verify width")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_profile(path: Path):
    data = json.loads(path.read_text())
    owners = np.asarray(data["owners"], dtype=np.int16)
    hot = np.zeros((len(ROUTED_LAYERS), NUM_EXPERTS), dtype=bool)
    for layer, experts in enumerate(data["hot_experts"]):
        hot[layer, np.asarray(experts, dtype=np.int64)] = True
    return owners, hot


def sample_steps(requests, width, concurrency, steps, rng):
    """Batches of `concurrency` requests contributing `width` consecutive tokens."""
    usable = [r for _, r in requests if r.shape[0] >= width]
    out = []
    for _ in range(steps):
        picks = rng.choice(len(usable), size=min(concurrency, len(usable)), replace=False)
        rows = []
        for i in picks:
            routes = usable[i]
            start = rng.integers(0, routes.shape[0] - width + 1)
            rows.append(routes[start : start + width])
        out.append(np.concatenate(rows))
    return out


def score(steps, owners, hot):
    n_layers = len(ROUTED_LAYERS)
    cold_only = np.zeros(len(steps))
    max_model = np.zeros(len(steps))
    hot_us = np.zeros((len(steps), n_layers))
    cold_us = np.zeros((len(steps), n_layers))
    for s, batch in enumerate(steps):
        for layer in range(n_layers):
            ids = np.unique(batch[:, layer, :])
            own = owners[layer, ids]
            is_hot = hot[layer, ids]
            h = np.bincount(own[is_hot], minlength=EP_SIZE)
            c = np.bincount(own[~is_hot], minlength=EP_SIZE)
            cold_only[s] += c.max()
            per_rank = np.maximum(T_HOT_US * h, T_COLD_US * c)
            max_model[s] += per_rank.max()
            hot_us[s, layer] = (T_HOT_US * h).max()
            cold_us[s, layer] = (T_COLD_US * c).max()
    return cold_only, max_model, hot_us, cold_us


def main() -> None:
    args = parse_args()
    requests = load_requests(args.trace_dir)
    rng = np.random.default_rng(args.seed)
    steps = sample_steps(
        requests, args.tokens_per_step, args.concurrency, args.steps, rng
    )
    report = {
        "tokens_per_step": args.tokens_per_step,
        "concurrency": args.concurrency,
        "steps": len(steps),
        "t_hot_us": T_HOT_US,
        "t_cold_us": T_COLD_US,
        "balance_cold_fraction": T_HOT_US / (T_HOT_US + T_COLD_US),
        "profiles": {},
    }
    for spec in args.profile:
        label, _, path = spec.partition("=")
        owners, hot = load_profile(Path(path))
        cold_only, max_model, hot_us, cold_us = score(steps, owners, hot)
        layer_hot = hot_us.mean(axis=0)
        layer_cold = cold_us.mean(axis=0)
        cold_bound = layer_cold > layer_hot
        report["profiles"][label] = {
            "planner_objective_cold_only": float(cold_only.mean()),
            "step_us_max_model": float(max_model.mean()),
            "step_us_if_cold_free": float(hot_us.sum(axis=1).mean()),
            "step_us_if_serial": float((hot_us + cold_us).sum(axis=1).mean()),
            "layers_cold_bound": int(cold_bound.sum()),
            "layers_hot_bound": int((~cold_bound).sum()),
            "mean_cold_over_hot_ratio": float((layer_cold / layer_hot).mean()),
            "median_cold_over_hot_ratio": float(np.median(layer_cold / layer_hot)),
            "min_cold_over_hot_ratio": float((layer_cold / layer_hot).min()),
            "max_cold_over_hot_ratio": float((layer_cold / layer_hot).max()),
            "wasted_hot_us_per_step": float(
                np.maximum(hot_us - cold_us, 0).sum(axis=1).mean()
            ),
        }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
