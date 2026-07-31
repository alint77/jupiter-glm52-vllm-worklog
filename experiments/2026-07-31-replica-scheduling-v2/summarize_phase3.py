#!/usr/bin/env python3
"""Paired statistics for the Phase 3 acceptance-free A/B.

Arms alternate within one job, so round `r` of each arm is a matched pair.
Reports the paired difference, which removes any slow drift over the job, plus
the unpaired means for reference.

With MTP off, mean TPOT is the step time directly - no acceptance correction is
needed or possible - so it is the headline. Output throughput is reported too,
but it is the noisier of the two.
"""

import argparse
import json
import math
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--seqs", type=int, required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


METRICS = {
    "mean_tpot_ms": "step time (lower is better)",
    "output_throughput": "output tok/s (higher is better)",
    "mean_itl_ms": "inter-token latency",
    "median_tpot_ms": "median step time",
}


def load_runs(
    result_dir: Path, job: str, seqs: int, spec: str
) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {"off": {}, "exact": {}}
    stem = f"phase3-{job}-s{seqs}-{spec}"
    pattern = re.compile(rf"^{re.escape(stem)}-(off|exact)-r(\d+)\.json$")
    for path in sorted(result_dir.glob(f"{stem}-*-r*.json")):
        match = pattern.match(path.name)
        if match is None:
            continue
        runs[match.group(1)][int(match.group(2))] = json.loads(path.read_text())
    return runs


def paired_t(differences: list[float]) -> tuple[float, float, float]:
    """Return mean, standard deviation and t statistic of paired differences."""
    n = len(differences)
    if n < 2:
        return (differences[0] if differences else 0.0), 0.0, 0.0
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    sd = math.sqrt(variance)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
    return mean, sd, t


def main() -> None:
    args = parse_args()
    runs = load_runs(args.result_dir, args.job, args.seqs, args.spec)
    rounds = sorted(set(runs["off"]) & set(runs["exact"]))
    if not rounds:
        raise SystemExit("no matched off/exact rounds found")

    summary: dict = {
        "job": args.job,
        "seqs": args.seqs,
        "spec": args.spec,
        "paired_rounds": rounds,
        "metrics": {},
    }
    print(
        f"job {args.job}, {args.seqs} seqs, spec {args.spec}, "
        f"{len(rounds)} matched rounds\n"
    )
    for metric, label in METRICS.items():
        off = [runs["off"][r][metric] for r in rounds]
        exact = [runs["exact"][r][metric] for r in rounds]
        # Percent change per pair, so the statistic is scale free.
        deltas = [100 * (e - o) / o for o, e in zip(off, exact)]
        mean, sd, t = paired_t(deltas)
        off_mean = sum(off) / len(off)
        exact_mean = sum(exact) / len(exact)
        summary["metrics"][metric] = {
            "label": label,
            "off_mean": off_mean,
            "exact_mean": exact_mean,
            "off_values": off,
            "exact_values": exact,
            "paired_delta_percent_mean": mean,
            "paired_delta_percent_sd": sd,
            "paired_t": t,
        }
        print(
            f"{metric:22s} off {off_mean:9.3f}  exact {exact_mean:9.3f}  "
            f"paired {mean:+7.2f}% (sd {sd:5.2f}, t {t:+6.2f})   {label}"
        )

    if args.output:
        args.output.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
