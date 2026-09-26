#!/usr/bin/env python3
"""Tabulate the placement A/B: TPOT and acceptance-normalised step time.

Placement changes which kernel reads an expert, not the math, but hot and
cold Marlin can round differently, so greedy outputs and DFlash acceptance
drift between arms. Step time (TPOT x tokens per step) removes that: it is
what placement actually changes. Spec counters are cumulative since server
start and include the 2-request warmup, which is <1% of the tokens.

    summarize_ab.py ab-<job>...
"""

import json
import re
import statistics
import sys
from pathlib import Path


def tokens_per_step(path: Path) -> float:
    values = {
        match.group(1): float(match.group(2))
        for match in re.finditer(r"spec_decode_num_(\w+)_total\{[^}]*\} (\S+)", path.read_text())
    }
    return (values["accepted_tokens"] + values["drafts"]) / values["drafts"]


def main() -> None:
    rows = []
    for directory in map(Path, sys.argv[1:]):
        for decode in sorted(directory.glob("decode-*.json")):
            tag = decode.stem.removeprefix("decode-")
            tpot = json.loads(decode.read_text())["mean_tpot_ms"]
            spec = directory / f"spec-{tag}.txt"
            tps = tokens_per_step(spec) if spec.exists() else float("nan")
            rows.append((directory.name, tag, tpot, tps, tpot * tps))
    print(f"{'job':<12}{'run':<12}{'TPOT ms':>9}{'tok/step':>10}{'step ms':>9}")
    for row in rows:
        print(f"{row[0]:<12}{row[1]:<12}{row[2]:>9.2f}{row[3]:>10.2f}{row[4]:>9.2f}")
    for arm in ("linear", "profile"):
        picked = [row for row in rows if row[1].endswith(arm)]
        if picked:
            print(
                f"{arm:<8} n={len(picked)} TPOT {statistics.mean(r[2] for r in picked):.2f} "
                f"tok/step {statistics.mean(r[3] for r in picked):.2f} "
                f"step {statistics.mean(r[4] for r in picked):.2f} ms"
            )


if __name__ == "__main__":
    main()
