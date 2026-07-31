#!/usr/bin/env python3
"""Summarize same-node replica A/B runs with acceptance-corrected step time."""

import argparse
import json
import re
import statistics
from pathlib import Path

PATTERN = re.compile(
    r"samenode-(?P<job>\d+)-(?P<assignment>off|greedy)-"
    r"c(?P<concurrency>[14])-r\d+\.json"
)
METRICS = (
    "output_throughput",
    "mean_tpot_ms",
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
)


def mean(values):
    return {
        "mean": statistics.mean(values),
        "sample_stddev": statistics.stdev(values),
        "values": values,
    }


def welch_t(control, candidate):
    control_variance = statistics.variance(control)
    candidate_variance = statistics.variance(candidate)
    control_term = control_variance / len(control)
    candidate_term = candidate_variance / len(candidate)
    standard_error = (control_term + candidate_term) ** 0.5
    degrees_of_freedom = (control_term + candidate_term) ** 2 / (
        control_term**2 / (len(control) - 1) + candidate_term**2 / (len(candidate) - 1)
    )
    return {
        "t": (statistics.mean(candidate) - statistics.mean(control)) / standard_error,
        "degrees_of_freedom": degrees_of_freedom,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jobs = set(args.jobs)
    grouped = {}
    for path in sorted(args.result_dir.glob("samenode-*-r*.json")):
        match = PATTERN.fullmatch(path.name)
        if match is None or match["job"] not in jobs:
            continue
        key = (match["assignment"], int(match["concurrency"]))
        grouped.setdefault(key, []).append(json.loads(path.read_text()))

    summary = {"jobs": sorted(jobs), "arms": {}, "deltas": {}}
    for (assignment, concurrency), runs in sorted(grouped.items()):
        if len(runs) != 3:
            raise ValueError(
                f"Expected three repeats for {(assignment, concurrency)}, "
                f"got {len(runs)}"
            )
        if any(run["completed"] != run["num_prompts"] or run["failed"] for run in runs):
            raise ValueError(f"Incomplete requests for {(assignment, concurrency)}")
        values = {metric: [float(run[metric]) for run in runs] for metric in METRICS}
        step_times = [
            tpot * accepted
            for tpot, accepted in zip(
                values["mean_tpot_ms"],
                values["spec_decode_acceptance_length"],
            )
        ]
        summary["arms"][f"{assignment}-c{concurrency}"] = {
            "metrics": {
                metric: mean(metric_values) for metric, metric_values in values.items()
            },
            "acceptance_corrected_step_time_ms": mean(step_times),
        }

    for concurrency in (1, 4):
        off = summary["arms"].get(f"off-c{concurrency}")
        greedy = summary["arms"].get(f"greedy-c{concurrency}")
        if off is None or greedy is None:
            continue
        off_tps = off["metrics"]["output_throughput"]["mean"]
        greedy_tps = greedy["metrics"]["output_throughput"]["mean"]
        off_step = off["acceptance_corrected_step_time_ms"]["mean"]
        greedy_step = greedy["acceptance_corrected_step_time_ms"]["mean"]
        off_tps_values = off["metrics"]["output_throughput"]["values"]
        greedy_tps_values = greedy["metrics"]["output_throughput"]["values"]
        off_step_values = off["acceptance_corrected_step_time_ms"]["values"]
        greedy_step_values = greedy["acceptance_corrected_step_time_ms"]["values"]
        summary["deltas"][f"c{concurrency}"] = {
            "output_throughput_percent": 100 * (greedy_tps / off_tps - 1),
            "output_throughput_welch": welch_t(
                off_tps_values,
                greedy_tps_values,
            ),
            "acceptance_corrected_step_time_percent": 100
            * (greedy_step / off_step - 1),
            "acceptance_corrected_step_time_welch": welch_t(
                off_step_values,
                greedy_step_values,
            ),
        }

    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["deltas"], indent=2))


if __name__ == "__main__":
    main()
