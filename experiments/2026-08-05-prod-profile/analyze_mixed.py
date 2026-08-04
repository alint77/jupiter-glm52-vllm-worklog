"""Split a capture's steps into pure-decode and prefill-bearing classes.

The `decode-c4` capture turned out to straddle the end of prefill (see README),
which makes it the right trace for the question the 16K benchmark raised: what
does a step cost when a chunked prefill shares it with decode, and does that
step still replay from a CUDA graph?
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

from analyze_step_budget import EMPTY_BUCKET, rank_from_path, rank_steps  # noqa: E402

SPLIT_MS = 200.0


def budget(steps: list[dict]) -> dict:
    buckets: dict[str, float] = collections.defaultdict(float)
    for step in steps:
        for name, value in step["shares"].items():
            buckets[name] += value
        buckets[EMPTY_BUCKET] += step["gpu_empty_ms"]
    return {name: value / len(steps) for name, value in buckets.items()}


def describe(name: str, steps: list[dict]) -> dict:
    if not steps:
        return {}
    shares = budget(steps)
    total = sum(shares.values())
    walls = sorted(step["wall_ms"] for step in steps)
    counts: dict[str, int] = collections.Counter()
    for step in steps:
        for key, value in step["counts"].items():
            counts[key] += value
    summary = {
        "steps": len(steps),
        "wall_ms_mean": statistics.fmean(walls),
        "wall_ms_median": walls[len(walls) // 2],
        "wall_ms_min": walls[0],
        "wall_ms_max": walls[-1],
        "gpu_busy_ms": statistics.fmean(s["gpu_busy_ms"] for s in steps),
        "gpu_empty_ms": statistics.fmean(s["gpu_empty_ms"] for s in steps),
        "budget": dict(sorted(shares.items(), key=lambda kv: -kv[1])),
        "budget_total": total,
        "kernel_counts_per_step": {k: v / len(steps) for k, v in counts.items()},
    }
    print(f"\n=== {name}: {len(steps)} rank-steps")
    print(
        f"  wall ms  mean {summary['wall_ms_mean']:8.2f}"
        f"  median {summary['wall_ms_median']:8.2f}"
        f"  min {summary['wall_ms_min']:7.2f}  max {summary['wall_ms_max']:8.2f}"
    )
    print(f"  gpu busy {summary['gpu_busy_ms']:8.2f}   empty {summary['gpu_empty_ms']:6.2f}")
    for key, value in summary["budget"].items():
        print(f"    {key:46} {value:9.3f} ms {value / total * 100:6.2f}%")
    print("  kernels per step:")
    for key, value in sorted(summary["kernel_counts_per_step"].items()):
        print(f"    {key:46} {value:8.1f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--split-ms", type=float, default=SPLIT_MS)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    traces = sorted(args.capture.glob("*.trace.json.gz"))
    per_rank = {rank_from_path(p): rank_steps(p, False) for p in traces}
    flat = [step for steps in per_rank.values() for step in steps]

    decode = [s for s in flat if s["wall_ms"] < args.split_ms]
    prefill = [s for s in flat if s["wall_ms"] >= args.split_ms]
    result = {
        "split_ms": args.split_ms,
        "decode_only": describe("decode-only steps", decode),
        "prefill_bearing": describe("prefill-bearing steps", prefill),
    }
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
