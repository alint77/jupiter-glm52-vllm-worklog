#!/usr/bin/env python3
"""Dissect communication kernels in matched four-rank decode traces."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
import statistics
from pathlib import Path

CUSTOM_AR = "cross_device_reduce_1stage"
NCCL_AG = "ncclDevKernel_AllGather"
RANK_RE = re.compile(r"_rank(\d+)\.")

CUSTOM_PHASES = (
    ("target", 157),
    ("mtp1-prep", 1),
    ("mtp1-forward", 2),
    ("mtp2-prep", 1),
    ("mtp2-forward", 2),
    ("mtp3-prep", 1),
    ("mtp3-forward", 2),
)
NCCL_PHASES = ("target-vocab", "mtp1-vocab", "mtp2-vocab", "mtp3-vocab")

# M=4 target batch, hidden_size=6144, BF16, four TP ranks.
CUSTOM_PAYLOAD_BYTES = 4 * 6144 * 2
CUSTOM_REMOTE_BYTES = 3 * CUSTOM_PAYLOAD_BYTES
# vocab_size=154880, four-way tensor-parallel vocabulary shard, BF16.
NCCL_LOCAL_BYTES = {
    "target-vocab": 4 * (154880 // 4) * 2,
    "mtp1-vocab": (154880 // 4) * 2,
    "mtp2-vocab": (154880 // 4) * 2,
    "mtp3-vocab": (154880 // 4) * 2,
}


def load_trace(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)["traceEvents"]


def rank_from_path(path: Path) -> int:
    match = RANK_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot find rank in {path}")
    return int(match.group(1))


def launch_correlations(events: list[dict], start: float, end: float) -> list[int]:
    launches = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") in {"cuda_runtime", "cuda_driver"}
        and start <= event["ts"] < end
        and (
            "LaunchKernel" in event["name"]
            or "launchKernel" in event["name"]
            or "GraphLaunch" in event["name"]
        )
    ]
    return [event["args"]["correlation"] for event in launches]


def step_comms(events: list[dict]) -> list[dict]:
    executes = sorted(
        (
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("cat") == "user_annotation"
            and event["name"].startswith("execute_")
        ),
        key=lambda event: event["ts"],
    )
    kernels_by_correlation: dict[int, list[dict]] = collections.defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None:
            kernels_by_correlation[correlation].append(event)

    steps = []
    for step, (current, following) in enumerate(zip(executes, executes[1:])):
        correlations = launch_correlations(
            events, current["ts"], following["ts"]
        )
        kernels = sorted(
            (
                kernel
                for correlation in correlations
                for kernel in kernels_by_correlation[correlation]
            ),
            key=lambda event: event["ts"],
        )
        custom = [event for event in kernels if CUSTOM_AR in event["name"]]
        nccl = [event for event in kernels if NCCL_AG in event["name"]]
        if len(custom) != 166 or len(nccl) != 4:
            raise ValueError(
                f"step {step}: expected 166 custom AR and four NCCL AG "
                f"kernels, found {len(custom)} and {len(nccl)}"
            )
        custom_rows = []
        offset = 0
        for phase, count in CUSTOM_PHASES:
            for ordinal, event in enumerate(custom[offset : offset + count]):
                custom_rows.append(
                    {
                        "phase": phase,
                        "ordinal": ordinal,
                        "duration_us": event["dur"],
                    }
                )
            offset += count
        nccl_rows = [
            {
                "phase": phase,
                "ordinal": 0,
                "duration_us": event["dur"],
            }
            for phase, event in zip(NCCL_PHASES, nccl)
        ]
        steps.append(
            {
                "step": step,
                "custom": custom_rows,
                "nccl": nccl_rows,
            }
        )
    return steps


def load_capture(path: Path) -> dict[int, list[dict]]:
    traces = sorted(path.glob("*.trace.json.gz"))
    if len(traces) != 4:
        raise ValueError(f"expected four rank traces in {path}, found {len(traces)}")
    return {rank_from_path(trace): step_comms(load_trace(trace)) for trace in traces}


def aligned_step(capture: dict[int, list[dict]], step: int) -> dict:
    result = {}
    for family in ("custom", "nccl"):
        rank_rows = [capture[rank][step][family] for rank in sorted(capture)]
        keys = [
            [(row["phase"], row["ordinal"]) for row in rows]
            for rows in rank_rows
        ]
        if any(key != keys[0] for key in keys[1:]):
            raise ValueError(f"unaligned {family} calls in step {step}")
        calls = []
        for index, (phase, ordinal) in enumerate(keys[0]):
            durations = [rows[index]["duration_us"] for rows in rank_rows]
            calls.append(
                {
                    "phase": phase,
                    "ordinal": ordinal,
                    "rank_duration_us": durations,
                    "observed_mean_us": statistics.mean(durations),
                    "floor_us": min(durations),
                }
            )
        result[family] = calls
    return result


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(captures: dict[str, dict[int, list[dict]]]) -> tuple[dict, list[dict]]:
    step_rows = []
    aligned = {}
    for capture_name, capture in captures.items():
        step_count = min(len(steps) for steps in capture.values())
        for step in range(step_count):
            data = aligned_step(capture, step)
            aligned[(capture_name, step)] = data
            custom_observed = sum(
                call["observed_mean_us"] for call in data["custom"]
            )
            custom_floor = sum(call["floor_us"] for call in data["custom"])
            nccl_observed = sum(
                call["observed_mean_us"] for call in data["nccl"]
            )
            nccl_floor = sum(call["floor_us"] for call in data["nccl"])
            step_rows.append(
                {
                    "capture": capture_name,
                    "step": step,
                    "custom_observed_ms": custom_observed / 1000,
                    "custom_floor_ms": custom_floor / 1000,
                    "nccl_observed_ms": nccl_observed / 1000,
                    "nccl_floor_ms": nccl_floor / 1000,
                    "total_observed_ms": (custom_observed + nccl_observed)
                    / 1000,
                    "total_floor_ms": (custom_floor + nccl_floor) / 1000,
                    "sync_excess_ms": (
                        custom_observed
                        + nccl_observed
                        - custom_floor
                        - nccl_floor
                    )
                    / 1000,
                }
            )

    steady_rows = [
        row
        for row in step_rows
        if not (row["capture"] == "1092954" and row["step"] == 0)
    ]
    phase_rows: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for (capture_name, step), families in aligned.items():
        if capture_name == "1092954" and step == 0:
            continue
        for family, calls in families.items():
            for call in calls:
                phase_rows[(family, call["phase"])].append(call)

    phases = {}
    for (family, phase), calls in phase_rows.items():
        step_count = len(steady_rows)
        phases[f"{family}:{phase}"] = {
            "calls_per_step": len(calls) / step_count,
            "observed_ms_per_step": sum(
                call["observed_mean_us"] for call in calls
            )
            / step_count
            / 1000,
            "floor_ms_per_step": sum(call["floor_us"] for call in calls)
            / step_count
            / 1000,
        }

    distributions = {}
    for family in ("custom", "nccl"):
        rank_durations = []
        floors = []
        for (capture_name, step), families in aligned.items():
            if capture_name == "1092954" and step == 0:
                continue
            for call in families[family]:
                rank_durations.extend(call["rank_duration_us"])
                floors.append(call["floor_us"])
        distributions[family] = {
            "rank_kernel_p50_us": percentile(rank_durations, 0.50),
            "rank_kernel_p90_us": percentile(rank_durations, 0.90),
            "rank_kernel_p99_us": percentile(rank_durations, 0.99),
            "rank_kernel_max_us": max(rank_durations),
            "aligned_floor_p50_us": percentile(floors, 0.50),
            "aligned_floor_p90_us": percentile(floors, 0.90),
            "aligned_floor_p99_us": percentile(floors, 0.99),
        }

    metric_keys = (
        "custom_observed_ms",
        "custom_floor_ms",
        "nccl_observed_ms",
        "nccl_floor_ms",
        "total_observed_ms",
        "total_floor_ms",
        "sync_excess_ms",
    )
    summary = {
        "steady_steps": len(steady_rows),
        "excluded": "1092954 step 0 (capture-start 16.872 ms AR outlier)",
        "means": {
            key: statistics.mean(row[key] for row in steady_rows)
            for key in metric_keys
        },
        "phases": phases,
        "distributions": distributions,
        "traffic_per_rank_per_step": {
            "custom_payload_bytes_per_call": CUSTOM_PAYLOAD_BYTES,
            "custom_remote_read_bytes": CUSTOM_REMOTE_BYTES * 166,
            "nccl_remote_receive_bytes": sum(
                3 * size for size in NCCL_LOCAL_BYTES.values()
            ),
            "total_remote_bytes": CUSTOM_REMOTE_BYTES * 166
            + sum(3 * size for size in NCCL_LOCAL_BYTES.values()),
        },
    }
    return summary, step_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            }
            for row in rows
        )


def print_report(summary: dict) -> None:
    means = summary["means"]
    print(f"steady steps: {summary['steady_steps']}")
    print(f"excluded: {summary['excluded']}")
    print(
        "custom AR: "
        f"{means['custom_observed_ms']:.6f} ms observed, "
        f"{means['custom_floor_ms']:.6f} ms floor"
    )
    print(
        "NCCL AG:   "
        f"{means['nccl_observed_ms']:.6f} ms observed, "
        f"{means['nccl_floor_ms']:.6f} ms floor"
    )
    print(
        "total:     "
        f"{means['total_observed_ms']:.6f} ms observed, "
        f"{means['total_floor_ms']:.6f} ms floor, "
        f"{means['sync_excess_ms']:.6f} ms synchronization excess"
    )
    print("\nphase                        calls  observed ms    floor ms")
    for name, row in summary["phases"].items():
        print(
            f"{name:28s} {row['calls_per_step']:5.0f} "
            f"{row['observed_ms_per_step']:12.6f} "
            f"{row['floor_ms_per_step']:11.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    captures = {
        path.parent.name.rsplit("-", 1)[-1]: load_capture(path)
        for path in args.captures
    }
    summary, rows = summarize(captures)
    print_report(summary)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
