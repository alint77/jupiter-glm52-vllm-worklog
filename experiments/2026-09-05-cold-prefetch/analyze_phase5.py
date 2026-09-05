# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired comparison of the two phase-5 arms on the identical question set.

Two independent accuracies at a few hundred questions cannot resolve a defect
worth catching: the interval on their difference is several points wide. The
arms answer the same questions, so the informative quantity is the discordant
pairs -- questions one arm got right and the other wrong.

McNemar's test asks whether that discordance is *asymmetric*. Run-to-run noise,
which phase 3b showed is substantial here and present in the baseline, produces
symmetric discordance. A staging defect produces asymmetry. That distinction is
what the test buys, and it is why no separate within-arm control is needed.
"""

import json
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).parent


def two_sided_exact(b: int, c: int) -> float:
    """Exact binomial p for b successes of b+c at p=0.5, two-sided."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def main() -> int:
    arms = {}
    for arm in ("baseline", "staged"):
        path = HERE / f"p5-{arm}-gsm8k.json"
        if not path.exists():
            print(f"missing {path.name}")
            return 1
        arms[arm] = json.loads(path.read_text())

    base, staged = arms["baseline"], arms["staged"]
    if base["labels"] != staged["labels"]:
        print("FAIL: arms did not answer the same questions")
        return 1

    n = len(base["correct"])
    print(f"questions: {n}")
    for arm, data in arms.items():
        print(
            f"  {arm:9s} accuracy {data['accuracy']:.4f}  "
            f"({sum(data['correct'])}/{n})  {data['seconds']:.0f}s"
        )

    # b: baseline right, staged wrong. c: staged right, baseline wrong.
    b = sum(1 for x, y in zip(base["correct"], staged["correct"]) if x and not y)
    c = sum(1 for x, y in zip(base["correct"], staged["correct"]) if y and not x)
    agree = n - b - c
    p = two_sided_exact(b, c)
    print(f"\nconcordant {agree}  baseline-only {b}  staged-only {c}")
    print(f"McNemar exact two-sided p = {p:.4f}")
    delta = staged["accuracy"] - base["accuracy"]
    print(f"accuracy delta {delta:+.4f}")

    if p < 0.05:
        print("\nASYMMETRIC: the arms differ beyond chance. Investigate.")
        return 1
    print(
        f"\nNo asymmetry detected (p={p:.3f}). {b + c} discordant pairs are "
        "consistent with this configuration's run-to-run noise."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
