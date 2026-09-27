#!/usr/bin/env python3
"""GLM-5.3 replica budgets, offline: the MiMo builder (mimo_replicas.py) on the
GLM routing capture, against the replicas the production profile already has.

Held-out 8-position windows stand in for MTP7 verify steps. Units are active
cold experts on the busiest rank, summed over layers per step (what exact
assignment minimises), not milliseconds.

    glm_replicas.py --trace-dir .../snap-1535650-a --profile glm53-w4a16-2496.json \
        --budgets 985 1500 2000 [--write-profile BUDGET OUT.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-routing-profile"))
from mimo_replicas import EP, active_mask, evaluate, load_steps, place, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--budgets", type=int, nargs="+", required=True)
    ap.add_argument("--write-profile", nargs=2, metavar=("BUDGET", "OUT"))
    args = ap.parse_args()
    profile = json.loads(args.profile.read_text())
    layers = profile["routed_layers"]
    owners = np.asarray(profile["owners"])
    hot = np.zeros(owners.shape, dtype=bool)
    for layer, ids in enumerate(profile["hot_experts"]):
        hot[layer, ids] = True
    n = owners.shape[1]
    train = active_mask(load_steps(args.trace_dir, "train", layers), n)
    held = active_mask(load_steps(args.trace_dir, "heldout", layers), n)
    res = {"steps": {"train": len(train), "heldout": len(held)},
           "none": evaluate(held, owners, hot, np.full(owners.shape, -1)),
           "profile": evaluate(held, owners, hot, np.asarray(profile["secondary_ranks"]))}
    gain = score(train, owners, hot)
    placed = {}
    for budget in args.budgets:
        placed[budget] = place(gain, owners, budget)
        res[str(budget)] = evaluate(held, owners, hot, placed[budget])
    print(json.dumps(res, indent=1))
    if args.write_profile:
        budget, out = int(args.write_profile[0]), Path(args.write_profile[1])
        out.write_text(json.dumps(dict(
            profile, secondary_ranks=placed[budget].tolist(),
            optimizer=profile["optimizer"] + f"+replicas-minmax-cold-{budget}")) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
