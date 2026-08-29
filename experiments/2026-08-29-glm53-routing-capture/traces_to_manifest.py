#!/usr/bin/env python3
"""Adapt a live routing-capture directory for ``optimize_routing_profile.py``.

The server writes ``manifest.jsonl`` (one record per request, keyed by
``request_id``); the optimizer reads ``manifest.json`` (a list keyed by
``request_hash``, optionally carrying a ``split``). This bridges the two and
assigns the train/held-out split deterministically, so re-running the optimizer
on the same traces reproduces the same profile.

Traces whose routes are the identity fallback (experts 0..7 on every layer) are
dropped: they come from steps where routed-expert return was not populated.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROUTED_LAYERS = tuple(range(3, 78))
NUM_EXPERTS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--min-positions", type=int, default=16)
    parser.add_argument(
        "--attribution",
        type=Path,
        help="requests.jsonl from the driver, mapping request ids to task ids",
    )
    return parser.parse_args()


def is_identity_trace(routes: np.ndarray) -> bool:
    default = np.arange(8)
    return bool(
        np.all(routes[:, ROUTED_LAYERS, :][:, :, :8] == default)
        if routes.shape[2] >= 8
        else False
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("Held-out fraction must lie strictly between 0 and 1")
    jsonl = args.trace_dir / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in jsonl.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No routing traces were recorded in {args.trace_dir}")

    domains = {}
    if args.attribution and args.attribution.is_file():
        for line in args.attribution.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                domains[entry["request_id"]] = entry["domain"]

    kept, dropped_identity, dropped_short, dropped_missing = [], 0, 0, 0
    for record in records:
        path = args.trace_dir / record["file"]
        if not path.is_file():
            dropped_missing += 1
            continue
        routes = np.load(path, mmap_mode="r")
        if routes.ndim != 3 or routes.shape[1] <= ROUTED_LAYERS[-1]:
            dropped_missing += 1
            continue
        if routes.shape[0] < args.min_positions:
            dropped_short += 1
            continue
        if is_identity_trace(routes):
            dropped_identity += 1
            continue
        kept.append((record, int(routes.shape[0])))

    if len(kept) < 2:
        raise ValueError(
            f"Only {len(kept)} usable traces; the optimizer needs a training "
            "split that still leaves held-out requests"
        )

    # Hash-based split so adding traces never reshuffles earlier assignments.
    entries = []
    for record, positions in kept:
        request_hash = hashlib.sha256(record["file"].encode()).hexdigest()[:16]
        bucket = int(request_hash[:8], 16) / 0xFFFFFFFF
        entries.append(
            {
                "file": record["file"],
                "request_hash": request_hash,
                "split": "heldout" if bucket < args.holdout_fraction else "train",
                "routed_positions": positions,
                "output_tokens": record.get("output_tokens"),
                **(
                    {"domain": domains[record["request_id"]]}
                    if record.get("request_id") in domains
                    else {}
                ),
            }
        )

    train = [entry for entry in entries if entry["split"] == "train"]
    heldout = [entry for entry in entries if entry["split"] == "heldout"]
    if not train or not heldout:
        raise ValueError("Split left one side empty; adjust --holdout-fraction")

    (args.trace_dir / "manifest.json").write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "traces_recorded": len(records),
                "traces_kept": len(entries),
                "train_requests": len(train),
                "heldout_requests": len(heldout),
                "routed_positions": sum(e["routed_positions"] for e in entries),
                "dropped_identity": dropped_identity,
                "dropped_short": dropped_short,
                "dropped_missing": dropped_missing,
                "domains": len({e["domain"] for e in entries if "domain" in e}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
