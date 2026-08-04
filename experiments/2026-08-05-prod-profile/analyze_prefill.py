"""Kernel-family budget for the prefill capture.

`analyze_step_budget.summarize` assumes a decode step: it reads the hot/cold
Marlin tier pair that only exists when `apply_tiered` takes the two-stream path,
which prefill does not (its token count exceeds `overlap_max_tokens`, so the
tiers run serially). This does the same launch-correlation attribution and
budget, without the tier statistics.
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


def summarize(arm_dir: Path) -> dict:
    traces = sorted(arm_dir.glob("*.trace.json.gz"))
    if len(traces) != 4:
        raise ValueError(f"expected four rank traces in {arm_dir}, found {traces}")
    per_rank = {rank_from_path(path): rank_steps(path, False) for path in traces}
    flat = [step for steps in per_rank.values() for step in steps]
    if not flat:
        raise ValueError(f"no steps found in {arm_dir}")

    buckets: dict[str, float] = collections.defaultdict(float)
    for step in flat:
        for name, value in step["shares"].items():
            buckets[name] += value
        buckets[EMPTY_BUCKET] += step["gpu_empty_ms"]
    budget = {name: value / len(flat) for name, value in buckets.items()}

    def per_step(key):
        return statistics.fmean(step[key] for step in flat)

    return {
        "rank_steps": len(flat),
        "steps_per_rank": {str(k): len(v) for k, v in per_rank.items()},
        "census": dict(
            collections.Counter(
                name for step in flat for name in step.get("census", {})
            )
        ),
        "wall_ms_per_step": per_step("wall_ms"),
        "gpu_span_ms_per_step": per_step("gpu_span_ms"),
        "gpu_busy_ms_per_step": per_step("gpu_busy_ms"),
        "gpu_empty_ms_per_step": per_step("gpu_empty_ms"),
        "budget": dict(sorted(budget.items(), key=lambda kv: -kv[1])),
        "budget_total": sum(budget.values()),
        "step_wall_ms": sorted(step["wall_ms"] for step in flat),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    summary = summarize(args.capture)

    print(f"=== {args.capture.name}")
    print(f"rank-steps {summary['rank_steps']}  per rank {summary['steps_per_rank']}")
    for key in (
        "wall_ms_per_step",
        "gpu_span_ms_per_step",
        "gpu_busy_ms_per_step",
        "gpu_empty_ms_per_step",
    ):
        print(f"  {key:34} {summary[key]:10.3f}")
    print("  additive budget:")
    for name, value in summary["budget"].items():
        share = value / summary["budget_total"] * 100
        print(f"    {name:46} {value:8.3f} ms {share:6.2f}%")
    print(f"    {'total':46} {summary['budget_total']:8.3f} ms")
    walls = summary["step_wall_ms"]
    print(f"  step wall ms: min {walls[0]:.2f} median {walls[len(walls) // 2]:.2f} max {walls[-1]:.2f}")

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
