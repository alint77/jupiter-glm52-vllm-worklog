#!/usr/bin/env python3
"""Why the hot/cold Marlin union is 0.70 of the sum rather than 0.50.

Each routed layer issues four Marlin launches: two chained GEMMs for the hot
tier on one stream, two for the cold tier on another. With the tight
shared-memory policy the two streams are co-resident, so the layer's union
should approach ``max(hot_chain, cold_chain)`` rather than their sum.

Two very different things could hold the ratio at 0.70:

  imperfect overlap   the tiers do not actually co-reside, so union tends
                      toward the sum. Fixing it means occupancy work - grid
                      shape, registers, shared memory.
  tier imbalance      the tiers overlap fine, but one is consistently longer,
                      so the union is bounded by the longer one. Fixing it
                      means moving experts between HBM and Grace residency,
                      which is a placement problem, not a kernel one.

They are separable: compare the measured union against the floor the observed
chain lengths allow. If measured is close to that floor, overlap is working
and imbalance is the whole story.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

from analyze_marlin_overlap import (
    GPU_CATEGORIES,
    MARLIN,
    RUNTIME_CATEGORIES,
    STEP_MARKER,
    TOPK_ANCHOR,
    load,
    union_us,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", nargs=2, action="append", metavar=("NAME", "DIR"), required=True
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def layer_rows(directory: Path, max_steps: int) -> list[dict]:
    rows = []
    for path in sorted(Path(directory).glob("*.pt.trace.json.gz")):
        events = load(path)
        by_correlation: dict[int, list[dict]] = collections.defaultdict(list)
        for event in events:
            correlation = event.get("args", {}).get("correlation")
            if event.get("cat") in GPU_CATEGORIES and correlation is not None:
                by_correlation[correlation].append(event)
        runtime = [e for e in events if e.get("cat") in RUNTIME_CATEGORIES]
        annotations = sorted(
            (
                e
                for e in events
                if e.get("cat") == "user_annotation" and STEP_MARKER in e["name"]
            ),
            key=lambda e: e["ts"],
        )
        for annotation in annotations[:max_steps]:
            end = annotation["ts"] + annotation["dur"]
            launches = [
                e
                for e in runtime
                if annotation["ts"] <= e["ts"] < end and "GraphLaunch" in e["name"]
            ]
            if len(launches) != 1:
                continue
            ops = by_correlation[launches[0]["args"]["correlation"]]
            anchors = sorted(
                (e for e in ops if TOPK_ANCHOR in e["name"]), key=lambda e: e["ts"]
            )
            graph_end = max(float(e["ts"] + e["dur"]) for e in ops)
            for index, anchor in enumerate(anchors):
                stop = (
                    float(anchors[index + 1]["ts"])
                    if index + 1 < len(anchors)
                    else graph_end
                )
                segment = [
                    e
                    for e in ops
                    if float(anchor["ts"]) <= float(e["ts"]) < stop
                    and MARLIN in e["name"]
                ]
                if len(segment) != 4:
                    continue
                by_stream: dict[int, list[dict]] = collections.defaultdict(list)
                for event in segment:
                    by_stream[int(event["args"].get("stream", -1))].append(event)
                if sorted(len(v) for v in by_stream.values()) != [2, 2]:
                    continue
                chains = []
                for events_on_stream in by_stream.values():
                    work = sum(float(e["dur"]) for e in events_on_stream)
                    start = min(float(e["ts"]) for e in events_on_stream)
                    finish = max(float(e["ts"] + e["dur"]) for e in events_on_stream)
                    chains.append({"work": work, "start": start, "finish": finish})
                chains.sort(key=lambda c: -c["work"])
                long_chain, short_chain = chains
                spans = [
                    (float(e["ts"]), float(e["ts"] + e["dur"])) for e in segment
                ]
                total = sum(float(e["dur"]) for e in segment)
                rows.append(
                    {
                        "union": union_us(spans),
                        "sum": total,
                        # Floor given the observed chain lengths: even perfect
                        # co-residency cannot beat the longer chain.
                        "floor": long_chain["work"],
                        # Floor if the two tiers were also equal in length.
                        "balanced_floor": total / 2.0,
                        "long_work": long_chain["work"],
                        "short_work": short_chain["work"],
                        "start_skew": abs(
                            long_chain["start"] - short_chain["start"]
                        ),
                    }
                )
    return rows


def summarize(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return statistics.mean(r[key] for r in rows)

    union, total = mean("union"), mean("sum")
    floor, balanced = mean("floor"), mean("balanced_floor")
    imbalance = [
        100 * (r["long_work"] - r["short_work"]) / r["sum"] for r in rows
    ]
    return {
        "layers": len(rows),
        "union_us": union,
        "sum_us": total,
        "union_over_sum": union / total,
        "overlap_floor_us": floor,
        "overlap_floor_ratio": floor / total,
        # How much of the gap to the floor is left unrealised by the kernels.
        "overlap_efficiency_us": union - floor,
        "balanced_floor_us": balanced,
        "recoverable_by_balance_us": floor - balanced,
        "tier_imbalance_percent": statistics.mean(imbalance),
        "start_skew_us": mean("start_skew"),
    }


def main() -> None:
    args = parse_args()
    results = {
        name: summarize(layer_rows(Path(directory), args.max_steps))
        for name, directory in args.arm
    }
    width = max(len(k) for r in results.values() for k in r)
    print(f"{'metric':{width}s}" + "".join(f"{n:>14s}" for n in results))
    for key in next(iter(results.values())):
        print(
            f"{key:{width}s}"
            + "".join(f"{results[n][key]:14.3f}" for n in results)
        )

    for name, row in results.items():
        per_step = 75 / 1000.0
        print(
            f"\n{name}: union {row['union_us']:.1f}us/layer against an overlap "
            f"floor of {row['overlap_floor_us']:.1f}us, so the kernels leave "
            f"{row['overlap_efficiency_us']:.1f}us/layer "
            f"({row['overlap_efficiency_us'] * per_step:.2f} ms/rank-step) "
            f"on the table.\n"
            f"{' ' * len(name)}  Perfect per-layer tier balance would reach "
            f"{row['balanced_floor_us']:.1f}us, a further "
            f"{row['recoverable_by_balance_us']:.1f}us/layer "
            f"({row['recoverable_by_balance_us'] * per_step:.2f} ms/rank-step). "
            f"Mean tier imbalance is {row['tier_imbalance_percent']:.1f}% of the "
            f"layer's Marlin work."
        )

    if args.output:
        args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
