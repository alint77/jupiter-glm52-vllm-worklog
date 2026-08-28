#!/usr/bin/env python3
"""Trim hot slots out of a placement profile to make room for a draft model.

The tiered planner budgets draft weights only when `method == "mtp"`, so the
4.58 GiB DFlash2 draft is invisible to it, and Phase 28 established that
raising the HBM reserve cannot substitute because the fail-closed audit's
`required_free` scales 1:1 with the planned reserve. The hot tier has to give
up the space instead.

**Which experts get dropped does not matter for what Phase 42 measures.**
Acceptance length is a property of the drafter and the target's weights; a
lower-residency configuration computes identical logits and accepts
identically, only slower. So this trims for safety and per-rank balance rather
than for value. A throughput arm would need a profile refitted from a routing
trace with `optimize_routing_profile.py`, not this truncation.

Invariants preserved, matching `tiered_moe_placement.py`:

  - `owners` untouched, so the EP-balance check still holds
  - `hot_experts[layer]` stays unique and in range
  - `secondary_ranks` untouched; replicas are charged to Grace, not HBM
    (`cold_slots = primary_slots - hot_slots + replica_slots`), so they cost
    nothing here and dropping them would only lose replica coverage
  - every rank sheds exactly the same number of slots
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# GLM-5.2 routed expert at W4G64: 3 x 2048 x 6144 params at ~4.5 bits.
EXPERT_MIB = 21.2


def trim(profile: dict, drop_per_rank: int) -> tuple[dict, dict]:
    owners = profile["owners"]
    hot = profile["hot_experts"]
    ep_size = profile["ep_size"]

    # Hot slots owned by each rank, grouped by layer.
    by_rank: dict[int, dict[int, list[int]]] = {r: defaultdict(list) for r in range(ep_size)}
    for layer_idx, (layer_owners, layer_hot) in enumerate(zip(owners, hot)):
        for expert_id in layer_hot:
            by_rank[layer_owners[expert_id]][layer_idx].append(expert_id)

    before = {r: sum(len(v) for v in by_rank[r].values()) for r in range(ep_size)}
    for rank, count in before.items():
        if drop_per_rank >= count:
            raise ValueError(
                f"rank {rank} holds {count} hot slots; cannot drop {drop_per_rank}"
            )

    # Spread the drop across layers in proportion to each layer's hot count, so
    # no layer is gutted. Largest-remainder apportionment, then take the
    # highest expert IDs in each layer for determinism.
    dropped: set[tuple[int, int]] = set()
    for rank in range(ep_size):
        layers = by_rank[rank]
        total = before[rank]
        quotas = {}
        remainders = []
        assigned = 0
        for layer_idx, ids in layers.items():
            exact = drop_per_rank * len(ids) / total
            quotas[layer_idx] = min(int(exact), len(ids))
            assigned += quotas[layer_idx]
            remainders.append((exact - int(exact), layer_idx))
        for _, layer_idx in sorted(remainders, reverse=True):
            if assigned >= drop_per_rank:
                break
            if quotas[layer_idx] < len(layers[layer_idx]):
                quotas[layer_idx] += 1
                assigned += 1
        if assigned != drop_per_rank:
            raise ValueError(f"rank {rank}: apportioned {assigned}, wanted {drop_per_rank}")
        for layer_idx, n in quotas.items():
            for expert_id in sorted(layers[layer_idx], reverse=True)[:n]:
                dropped.add((layer_idx, expert_id))

    out = dict(profile)
    out["hot_experts"] = [
        sorted(e for e in layer_hot if (layer_idx, e) not in dropped)
        for layer_idx, layer_hot in enumerate(hot)
    ]
    out["optimizer"] = (
        f"{profile['optimizer']}+trimmed-{drop_per_rank}-per-rank-for-draft-residency"
    )

    after: dict[int, int] = defaultdict(int)
    for layer_idx, (layer_owners, layer_hot) in enumerate(zip(owners, out["hot_experts"])):
        for expert_id in layer_hot:
            after[layer_owners[expert_id]] += 1
    report = {
        "hot_slots_by_rank_before": before,
        "hot_slots_by_rank_after": dict(after),
        "dropped_per_rank": drop_per_rank,
        "freed_mib_per_rank": round(drop_per_rank * EXPERT_MIB, 1),
        "freed_gib_per_rank": round(drop_per_rank * EXPERT_MIB / 1024, 3),
        "layers_touched": len({layer for layer, _ in dropped}),
    }
    return out, report


def validate(profile: dict) -> None:
    """Replay the checks in tiered_moe_placement.load_tiered_moe_placement_profile."""
    ep_size = profile["ep_size"]
    num_experts = profile["num_experts"]
    experts_per_rank, remainder = divmod(num_experts, ep_size)
    assert not remainder, "experts must divide evenly across ranks"

    n = len(profile["routed_layers"])
    for key in ("owners", "hot_experts", "secondary_ranks"):
        assert len(profile[key]) == n, f"{key} must have one row per layer"

    for i, (own, hot, sec) in enumerate(
        zip(profile["owners"], profile["hot_experts"], profile["secondary_ranks"])
    ):
        assert len(own) == num_experts, f"owners row {i} length"
        assert all(isinstance(o, int) and 0 <= o < ep_size for o in own), f"owner rank {i}"
        assert all(own.count(r) == experts_per_rank for r in range(ep_size)), (
            f"owners row {i} is not EP balanced"
        )
        assert all(isinstance(e, int) and 0 <= e < num_experts for e in hot), f"hot id {i}"
        assert len(set(hot)) == len(hot), f"hot experts not unique in layer {i}"
        assert len(sec) == num_experts, f"secondary row {i} length"
        assert all(isinstance(r, int) and -1 <= r < ep_size for r in sec), f"secondary {i}"
        assert not any(r == own[e] for e, r in enumerate(sec) if r >= 0), (
            f"secondary rank matches owner in layer {i}"
        )

    assert profile["training_request_hashes"] and profile["heldout_request_hashes"]
    assert not set(profile["training_request_hashes"]) & set(
        profile["heldout_request_hashes"]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--drop-per-rank",
        type=int,
        default=200,
        help="hot slots to remove from every rank; 200 frees ~4.1 GiB/GPU",
    )
    args = ap.parse_args()

    profile = json.loads(args.input.read_text())
    validate(profile)
    trimmed, report = trim(profile, args.drop_per_rank)
    validate(trimmed)

    args.output.write_text(json.dumps(trimmed))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
