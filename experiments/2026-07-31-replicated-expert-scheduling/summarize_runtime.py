#!/usr/bin/env python3
"""Summarize matched primary-only and greedy runtime jobs."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

RESULT_PATTERN = re.compile(
    r"runtime-(?P<job>\d+)-(?P<assignment>off|greedy)-"
    r"c(?P<concurrency>[14])-r\d+\.json"
)
METRICS = (
    "output_throughput",
    "total_token_throughput",
    "request_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def vram_summary(path: Path) -> dict[str, dict[str, int]]:
    by_gpu: dict[str, list[tuple[int, int]]] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            gpu = row[1].strip()
            used = int(row[3])
            free = int(row[4])
            by_gpu.setdefault(gpu, []).append((used, free))
    return {
        gpu: {
            "maximum_used_mib": max(used for used, _ in samples),
            "minimum_free_mib": min(free for _, free in samples),
        }
        for gpu, samples in sorted(by_gpu.items())
    }


def main() -> None:
    args = parse_args()
    jobs = set(args.jobs)
    grouped: dict[tuple[str, int], list[dict]] = {}
    tags: dict[tuple[str, int], str] = {}
    for path in sorted(args.result_dir.glob("runtime-*-r*.json")):
        match = RESULT_PATTERN.fullmatch(path.name)
        if match is None or match["job"] not in jobs:
            continue
        key = (match["assignment"], int(match["concurrency"]))
        grouped.setdefault(key, []).append(json.loads(path.read_text()))
        tags[key] = path.name.rsplit("-r", 1)[0]

    expected = {
        (assignment, concurrency)
        for assignment in ("off", "greedy")
        for concurrency in (1, 4)
    }
    if set(grouped) != expected:
        raise ValueError(f"Incomplete result groups: {set(grouped)}")

    summary = {"jobs": sorted(jobs), "arms": {}, "deltas": {}}
    for key, runs in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        assignment, concurrency = key
        if len(runs) != 2:
            raise ValueError(f"Expected two repeats for {key}, got {len(runs)}")
        name = f"{assignment}-c{concurrency}"
        metrics = {}
        for metric in METRICS:
            values = [float(run[metric]) for run in runs]
            metrics[metric] = {
                "mean": statistics.mean(values),
                "sample_stddev": statistics.stdev(values),
                "values": values,
            }
        if any(run["completed"] != run["num_prompts"] or run["failed"] for run in runs):
            raise ValueError(f"Incomplete requests in {name}")
        vram_path = args.result_dir / f"{tags[key]}-vram.csv"
        summary["arms"][name] = {
            "completed_per_repeat": [run["completed"] for run in runs],
            "metrics": metrics,
            "vram": vram_summary(vram_path),
        }

    for concurrency in (1, 4):
        control = summary["arms"][f"off-c{concurrency}"]["metrics"]
        candidate = summary["arms"][f"greedy-c{concurrency}"]["metrics"]
        control_tps = control["output_throughput"]["mean"]
        candidate_tps = candidate["output_throughput"]["mean"]
        summary["deltas"][f"c{concurrency}"] = {
            "output_throughput_percent": 100 * (candidate_tps / control_tps - 1),
            "mean_tpot_percent": 100
            * (candidate["mean_tpot_ms"]["mean"] / control["mean_tpot_ms"]["mean"] - 1),
            "acceptance_rate_points": (
                candidate["spec_decode_acceptance_rate"]["mean"]
                - control["spec_decode_acceptance_rate"]["mean"]
            ),
        }

    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["deltas"], indent=2))


if __name__ == "__main__":
    main()
