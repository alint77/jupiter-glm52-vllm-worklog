#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
)
CASES = (
    "control",
    "current-owners-frequency",
    "balanced-owners-frequency",
    "balanced-owners-tail",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def min_vram_headroom(path: Path) -> dict[str, float]:
    free_by_gpu = defaultdict(list)
    with path.open() as file:
        for row in csv.reader(file):
            free_by_gpu[int(row[1])].append(float(row[4]))
    per_gpu = {
        str(gpu): min(samples) for gpu, samples in sorted(free_by_gpu.items())
    }
    return {
        "minimum_mib": min(per_gpu.values()),
        "per_gpu_minimum_mib": per_gpu,
    }


def summarize_case(result_dir: Path, prefix: str) -> dict:
    repetitions = [
        json.loads((result_dir / f"{prefix}-r{repeat}.json").read_text())
        for repeat in (1, 2)
    ]
    invalid = any(
        result["completed"] != 24 or result["failed"] != 0
        for result in repetitions
    )
    if invalid:
        raise ValueError(f"{prefix} did not complete 24 requests in both repetitions")
    summary = {
        metric: {
            "mean": statistics.mean(result[metric] for result in repetitions),
            "range": [
                min(result[metric] for result in repetitions),
                max(result[metric] for result in repetitions),
            ],
        }
        for metric in METRICS
    }
    summary["completed_per_repetition"] = 24
    summary["failed_per_repetition"] = 0
    summary["vram_headroom"] = min_vram_headroom(
        result_dir / f"{prefix}-vram-timeseries.csv"
    )
    return summary


def main() -> None:
    args = parse_args()
    report = {}
    for concurrency, label_prefix in (("c1", ""), ("c4_dcp4", "c4-")):
        cases = {
            case: summarize_case(args.result_dir, f"{label_prefix}{case}")
            for case in CASES
        }
        control_tps = cases["control"]["output_throughput"]["mean"]
        for case in cases.values():
            throughput = case["output_throughput"]["mean"]
            case["output_throughput"]["delta_vs_control_percent"] = (
                100 * (throughput / control_tps - 1)
            )
        report[concurrency] = cases
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
