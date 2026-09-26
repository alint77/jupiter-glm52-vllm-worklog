#!/usr/bin/env python3
"""Per-decode-step tier load of held-out MiMo traces under two placements.

A DFlash k=7 step verifies 8 tokens, and each tier's Marlin reads every
distinct expert it is routed once per step, so the cost driver is distinct
experts per (step, layer, rank), not route hits. The trace records every
verified position, rejected drafts included (80 output tokens -> 144 rows,
18 steps), and the writer trims the ragged prefix, so each 8-row window is
one real verify step.

For each placement this reports, per step and rank averaged over layers:
distinct hot and cold experts, the cold share of expert bytes read (DAK's
offload ratio; its balance point is B_c2c / (B_c2c + B_hbm)), and the
critical rank's cold count summed over layers, which is what bounds a step.

    step_tiers.py --trace-dir DIR --profile P.json [--hot-slots N]
"""

import argparse
import json
from pathlib import Path

import numpy as np

STEP = 8
EP = 4


def load(trace_dir: Path, split: str, layers: tuple[int, ...]) -> list[np.ndarray]:
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    out = []
    for record in manifest:
        if record.get("split") != split:
            continue
        routes = np.load(trace_dir / record["file"])[:, list(layers), :]
        usable = routes.shape[0] // STEP * STEP
        out.append(routes[:usable].reshape(-1, STEP, len(layers), routes.shape[2]))
    return out


def tier_load(steps: np.ndarray, owners: np.ndarray, hot: np.ndarray) -> dict:
    n_steps, _, n_layers, _ = steps.shape
    num_experts = owners.shape[1]
    present = np.zeros((n_steps, n_layers, num_experts), dtype=bool)
    for layer in range(n_layers):
        ids = steps[:, :, layer, :].reshape(n_steps, -1)
        np.put_along_axis(present[:, layer, :], ids, True, axis=1)
    hot_d = np.zeros((n_steps, n_layers, EP))
    cold_d = np.zeros((n_steps, n_layers, EP))
    for rank in range(EP):
        owned = owners == rank
        hot_d[:, :, rank] = (present & (owned & hot)[None]).sum(axis=2)
        cold_d[:, :, rank] = (present & (owned & ~hot)[None]).sum(axis=2)
    total = hot_d + cold_d
    return {
        "steps": int(n_steps),
        "hot_distinct_per_layer_rank": float(hot_d.mean()),
        "cold_distinct_per_layer_rank": float(cold_d.mean()),
        "cold_byte_share": float(cold_d.sum() / total.sum()),
        "critical_cold_per_step": float(cold_d.max(axis=2).sum(axis=1).mean()),
        "critical_hot_per_step": float(hot_d.max(axis=2).sum(axis=1).mean()),
    }


def linear_even(num_experts: int, n_layers: int, hot_slots: int):
    per_rank = num_experts // EP
    owners = np.tile(np.repeat(np.arange(EP), per_rank), (n_layers, 1))
    hot = np.zeros_like(owners, dtype=bool)
    base, extra = divmod(hot_slots, n_layers)
    for layer in range(n_layers):
        count = base + (layer < extra)
        for rank in range(EP):
            owned = np.flatnonzero(owners[layer] == rank)
            start = layer % len(owned)
            hot[layer, np.roll(owned, -start)[:count]] = True
    return owners, hot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--hot-slots", type=int)
    parser.add_argument("--split", default="heldout")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    layers = tuple(profile["routed_layers"])
    num_experts = profile["num_experts"]
    owners = np.asarray(profile["owners"])
    hot = np.zeros_like(owners, dtype=bool)
    for layer, ids in enumerate(profile["hot_experts"]):
        hot[layer, ids] = True
    slots = args.hot_slots or int(hot.sum()) // EP
    steps = np.concatenate(load(args.trace_dir, args.split, layers))
    result = {
        "split": args.split,
        "hot_slots_per_rank": slots,
        "linear_even": tier_load(steps, *linear_even(num_experts, len(layers), slots)),
        "profile": tier_load(steps, owners, hot),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
