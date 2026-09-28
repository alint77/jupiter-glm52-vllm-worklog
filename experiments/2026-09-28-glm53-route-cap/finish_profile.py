#!/usr/bin/env python3
"""Order a frequency profile's hot lists and add Grace replicas, as for MiMo.

Each layer's hot list is sorted by training-split activation rate, so when the
planner has fewer slots than the profile lists it demotes the least-used
experts first. Replicas come from ../2026-09-26-mimo-routing-profile/
mimo_replicas.py (greedy on the training split, min-max cold). Held-out numbers
are printed against the currently served profile at the same slot count.

    finish_profile.py --trace-dir MERGED --profile profile-3239.json \
        --replicas 2000 --out glm53-w4a16-agentic-3239-r2000.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26-mimo-routing-profile"))
from mimo_replicas import active_mask, evaluate, load_steps, place, score  # noqa: E402

SERVED = HERE.parents[1] / "profiles/glm53-w4a16-2496.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--replicas", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = json.loads(args.profile.read_text())
    layers = p["routed_layers"]
    owners = np.asarray(p["owners"])
    num = owners.shape[1]
    train = active_mask(load_steps(args.trace_dir, "train", layers), num)
    held = active_mask(load_steps(args.trace_dir, "heldout", layers), num)
    freq = train.mean(0)
    hot = np.zeros(owners.shape, dtype=bool)
    for li, ids in enumerate(p["hot_experts"]):
        hot[li, ids] = True
    ordered = [sorted(ids, key=lambda e, li=li: -freq[li, e]) for li, ids in enumerate(p["hot_experts"])]
    secondary = place(score(train, owners, hot), owners, args.replicas)
    out = dict(p, hot_experts=[[int(e) for e in ids] for ids in ordered],
               secondary_ranks=secondary.tolist(),
               optimizer=p["optimizer"] + f"+frequency-ordered+replicas-minmax-cold-{args.replicas}")
    args.out.write_text(json.dumps(out) + "\n")
    served = json.loads(SERVED.read_text())
    s_hot = np.zeros(owners.shape, dtype=bool)
    for li, ids in enumerate(served["hot_experts"]):
        s_hot[li, ids] = True
    print(f"wrote {args.out}: hot/GPU {[int((hot & (owners == r)).sum()) for r in range(4)]}, "
          f"replicas/GPU {[int((secondary == r).sum()) for r in range(4)]}")
    print(f"held-out steps {len(held)}, train {len(train)}")
    for name, h, sec in (("served profile (2496 listed)", s_hot, np.asarray(served["secondary_ranks"])),
                         ("new profile", hot, secondary)):
        sec = sec.copy()
        sec[h] = -1
        ev = evaluate(held, owners, h, sec)
        print(f"  {name:30s} hot/GPU {h.sum() / 4:.0f}: mean cold/GPU/step "
              f"{ev['mean_cold_per_rank_per_step']:.1f}, busiest cold/step {ev['critical_cold_per_step']:.1f}")


if __name__ == "__main__":
    main()
