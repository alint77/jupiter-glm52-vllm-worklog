#!/usr/bin/env python3
"""Per-phase kernel and communication budget for the MTP3 c=1 captures.

Attribution is the launch-correlation scheme from
`2026-07-29-marlin-smem-monopoly/analyze_step_budget.py`: a kernel belongs to
the step whose CPU launch it came from, not to the step whose annotation
window its GPU timestamps happen to fall in. That file's `summarize` is not
reused, for two reasons this capture makes unavoidable:

* Its strict census asserts `EXPECT_COUNTS`, which describes GLM-5.2 at DCP4.
  This is GLM-5.3 at DCP1, so the census is reported, never asserted.
* It averages per-layer and CUDA-graph statistics unconditionally. Prefill
  runs eager, so those sequences are empty and `fmean` raises. Every such
  statistic here is guarded and reported as null when the phase has none.

The added table is the communication split. The shared bucket map lumps custom
all-reduce, NCCL all-gather and DCP NCCL into one line, but they have
different fixes, so comms kernels are also reported individually by name.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

SMEM_DIR = Path(__file__).resolve().parents[1] / "2026-07-29-marlin-smem-monopoly"
sys.path.insert(0, str(SMEM_DIR))

from analyze_step_budget import (  # noqa: E402
    BUDGET_BUCKETS,
    EMPTY_BUCKET,
    GPU_CATEGORIES,
    LAUNCH_CATEGORIES,
    load_trace,
    partition,
    rank_from_path,
    step_windows,
)
from analyze_profile_ab import family  # noqa: E402

COMMS_NEEDLES = (
    "cross_device_reduce",
    "ncclDevKernel",
    "nccl",
    "symm_mem",
    "multimem",
    "AllToAll",
    "SendRecv",
)


def is_comms(name: str) -> bool:
    return any(needle in name for needle in COMMS_NEEDLES)


def rank_analysis(path: Path) -> dict:
    """One load of one rank's trace; step budget rows plus kernel tallies."""
    events = load_trace(path)
    by_correlation: dict[int, list[dict]] = collections.defaultdict(list)
    for event in events:
        if event.get("cat") not in GPU_CATEGORIES:
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None:
            by_correlation[correlation].append(event)

    launches = sorted(
        (e for e in events if e.get("cat") in LAUNCH_CATEGORIES),
        key=lambda e: e["t"],
    )

    steps = []
    tally: dict[str, dict] = collections.defaultdict(
        lambda: {"count": 0, "us": 0.0}
    )
    for start, end in step_windows(events):
        window = [
            e["args"]["correlation"]
            for e in launches
            if start <= e["t"] < end and "correlation" in e.get("args", {})
        ]
        ops = sorted(
            (k for c in window for k in by_correlation[c]), key=lambda e: e["t"]
        )
        if not ops:
            continue
        for event in ops:
            row = tally[event["name"]]
            row["count"] += 1
            row["us"] += event["dur"]
        shares, busy, span, span_start = partition(ops)
        graph = [
            e for e in launches
            if start <= e["t"] < end and "GraphLaunch" in e["name"]
        ]
        graph_span = None
        if graph:
            target = by_correlation[
                min(graph, key=lambda e: e["t"])["args"]["correlation"]
            ]
            if target:
                graph_span = max(
                    e["t"] + e["dur"] / 1000 for e in target
                ) - min(e["t"] for e in target)
        steps.append(
            {
                "wall_ms": end - start,
                "gpu_span_ms": span,
                "gpu_busy_ms": busy,
                "gpu_empty_ms": span - busy,
                "lead_in_ms": span_start - start,
                "graph_span_ms": graph_span,
                "shares": shares,
            }
        )
    return {"steps": steps, "tally": {k: dict(v) for k, v in tally.items()}}


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_phase(phase_dir: Path) -> dict:
    traces = sorted(phase_dir.glob("*.trace.json.gz"))
    if not traces:
        raise FileNotFoundError(f"no rank traces in {phase_dir}")
    per_rank = {rank_from_path(p): rank_analysis(p) for p in traces}

    flat = [s for r in per_rank.values() for s in r["steps"]]
    if not flat:
        raise ValueError(f"{phase_dir}: no attributed steps -- capture window "
                         f"likely closed before any engine step ran")
    total = len(flat)

    buckets: dict[str, float] = collections.defaultdict(float)
    for step in flat:
        for name, value in step["shares"].items():
            buckets[name] += value
        buckets[EMPTY_BUCKET] += step["gpu_empty_ms"]
    budget = {k: v / total for k, v in buckets.items()}

    tally: dict[str, dict] = collections.defaultdict(
        lambda: {"count": 0, "us": 0.0}
    )
    for rank in per_rank.values():
        for name, row in rank["tally"].items():
            tally[name]["count"] += row["count"]
            tally[name]["us"] += row["us"]

    kernels = sorted(
        (
            {
                "kernel": name[:96],
                "family": family(name),
                "count_per_step": row["count"] / total,
                "ms_per_step": row["us"] / 1000 / total,
            }
            for name, row in tally.items()
        ),
        key=lambda r: -r["ms_per_step"],
    )
    comms = [row for row in kernels if is_comms(row["kernel"])]

    return {
        "phase_dir": str(phase_dir),
        "ranks": {str(r): len(v["steps"]) for r, v in per_rank.items()},
        "rank_steps_total": total,
        "steps_are_graph_replayed": sum(
            1 for s in flat if s["graph_span_ms"] is not None
        ) / total,
        "wall_ms_per_step": statistics.fmean(s["wall_ms"] for s in flat),
        "gpu_span_ms_per_step": statistics.fmean(s["gpu_span_ms"] for s in flat),
        "gpu_busy_ms_per_step": statistics.fmean(s["gpu_busy_ms"] for s in flat),
        "gpu_empty_ms_per_step": statistics.fmean(s["gpu_empty_ms"] for s in flat),
        "lead_in_ms_per_step": statistics.fmean(s["lead_in_ms"] for s in flat),
        "graph_span_ms_per_step": mean_or_none(
            [s["graph_span_ms"] for s in flat if s["graph_span_ms"] is not None]
        ),
        "budget_ms_per_step": dict(sorted(budget.items(), key=lambda i: -i[1])),
        "budget_total_ms_per_step": sum(budget.values()),
        "comms_kernels": comms,
        "comms_ms_per_step": sum(r["ms_per_step"] for r in comms),
        "top_kernels": kernels[:30],
        "distinct_kernels": len(kernels),
    }


def print_phase(label: str, s: dict, top: int) -> None:
    print(f"\n=== {label}: {s['phase_dir']}")
    print(f"rank-steps {s['rank_steps_total']} {s['ranks']}  "
          f"graph-replayed {100 * s['steps_are_graph_replayed']:.0f}%  "
          f"distinct kernels {s['distinct_kernels']}")
    for key in ("wall_ms_per_step", "gpu_span_ms_per_step", "gpu_busy_ms_per_step",
                "gpu_empty_ms_per_step", "lead_in_ms_per_step",
                "graph_span_ms_per_step"):
        value = s[key]
        print(f"  {key:26s} {'--' if value is None else f'{value:9.3f}'}")

    total = s["budget_total_ms_per_step"]
    print("  additive budget (ms/step, families sharing a segment split it):")
    for name, value in s["budget_ms_per_step"].items():
        print(f"    {name:44s} {value:8.3f}  {100 * value / total:5.2f}%")
    print(f"    {'total':44s} {total:8.3f}")

    print(f"  communication kernels ({s['comms_ms_per_step']:.3f} ms/step, "
          f"{100 * s['comms_ms_per_step'] / total:.2f}% of budget):")
    for row in s["comms_kernels"][:12]:
        print(f"    {row['kernel'][:64]:64s} {row['count_per_step']:7.1f}x "
              f"{row['ms_per_step']:8.3f} ms")

    print(f"  top {top} kernels by GPU time:")
    for row in s["top_kernels"][:top]:
        print(f"    {row['kernel'][:64]:64s} {row['count_per_step']:7.1f}x "
              f"{row['ms_per_step']:8.3f} ms  [{row['family']}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--labels", nargs="*", default=["prefill", "decode"])
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    results = {}
    for label in args.labels:
        directory = args.trace_root / label
        if not directory.is_dir():
            print(f"skip {label}: {directory} does not exist")
            continue
        try:
            summary = summarize_phase(directory)
        except (ValueError, FileNotFoundError) as error:
            print(f"{label}: {error}")
            continue
        results[label] = summary
        print_phase(label, summary, args.top)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=float) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
