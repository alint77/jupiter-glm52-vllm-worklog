#!/usr/bin/env python3
"""Compare unique-owner and static-secondary serving results."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("secondary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    control = json.loads(args.control.read_text())
    secondary = json.loads(args.secondary.read_text())
    for name, result in (("control", control), ("secondary", secondary)):
        if result["failed"] or result["completed"] != result["num_prompts"]:
            raise ValueError(f"{name} did not complete every request")
    matches = [
        left == right
        for left, right in zip(control["generated_texts"], secondary["generated_texts"])
    ]
    report = {
        "prompt_count": len(matches),
        "exact_text_matches": sum(matches),
        "all_exact": all(matches),
        "exact_match_required": args.require_exact,
        "control_output_tps": control["output_throughput"],
        "secondary_output_tps": secondary["output_throughput"],
        "control_acceptance_percent": control.get("spec_decode_acceptance_rate"),
        "secondary_acceptance_percent": secondary.get("spec_decode_acceptance_rate"),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and not report["all_exact"]:
        raise ValueError("Static-secondary output differs from unique ownership")


if __name__ == "__main__":
    main()
