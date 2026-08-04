"""Per-phase step budget for the production profile captures.

Reuses the launch-correlation attribution from
`2026-07-29-marlin-smem-monopoly/analyze_step_budget.py`, which assigns every
GPU kernel to an engine step through its CUDA launch correlation rather than by
comparing GPU timestamps against CPU annotation boundaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SMEM_DIR = (
    Path(__file__).resolve().parents[1] / "2026-07-29-marlin-smem-monopoly"
)
sys.path.insert(0, str(SMEM_DIR))

from analyze_step_budget import print_arm, summarize  # noqa: E402

LABELS = ["prefill-c1", "decode-c1", "decode-c4", "mixed-c4"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--labels", nargs="*", default=LABELS)
    args = parser.parse_args()

    results = {}
    for label in args.labels:
        directory = args.trace_root / label
        if not directory.is_dir():
            print(f"skip {label}: no directory")
            continue
        # Only the c1 decode capture matches the strict MTP3 decode census; the
        # others mix prefill chunks or concurrent sequences into a step.
        strict = label == "decode-c1"
        try:
            summary = summarize(directory, strict)
        except ValueError as error:
            print(f"{label}: strict census failed ({error}); retrying loose")
            summary = summarize(directory, False)
        results[label] = summary
        print_arm(label, summary)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
