"""Summarize the matched c1/q4 server A/B for the tight-shared-memory launch.

Reads the two measured repetitions of each arm and reports the mean plus the
per-repetition range, so a difference smaller than the spread is visible as
such rather than quoted as a result.
"""

import json
import pathlib
import sys

D = pathlib.Path(__file__).resolve().parent.parent
FIELDS = [
    ("output_throughput", "output tok/s", 2, False),
    ("mean_tpot_ms", "TPOT ms", 3, True),
    ("mean_ttft_ms", "TTFT ms", 1, True),
    ("mean_itl_ms", "ITL ms", 3, True),
]


def load(arm):
    reps = []
    for r in (1, 2):
        p = D / f"{arm}-r{r}.json"
        if p.exists():
            reps.append(json.loads(p.read_text()))
    return reps


def main():
    arms = {a: load(a) for a in ("off", "on")}
    missing = [a for a, r in arms.items() if len(r) < 2]
    if missing:
        print(f"incomplete: {missing} (have "
              f"{ {a: len(r) for a, r in arms.items()} })")
        if not all(arms.values()):
            sys.exit(1)

    print(f"{'metric':<14} {'off (control)':>22} {'on (tight smem)':>22} "
          f"{'delta':>9}")
    for key, label, prec, lower_better in FIELDS:
        vals = {}
        for arm, reps in arms.items():
            xs = [r[key] for r in reps if key in r]
            if not xs:
                break
            vals[arm] = (sum(xs) / len(xs), min(xs), max(xs))
        if len(vals) < 2:
            continue
        o, n = vals["off"], vals["on"]
        delta = n[0] / o[0] - 1 if o[0] else 0.0
        if lower_better:
            delta = -delta
        print(f"{label:<14} "
              f"{o[0]:>12.{prec}f} [{o[1]:.{prec}f},{o[2]:.{prec}f}]".ljust(37)
              + f"{n[0]:>12.{prec}f} [{n[1]:.{prec}f},{n[2]:.{prec}f}]".ljust(23)
              + f"{delta:>+8.2%}")

    # a delta inside the union of the two arms' ranges is not a result
    for key, label, _, _ in FIELDS:
        xs = [r[key] for r in arms["off"] if key in r]
        ys = [r[key] for r in arms["on"] if key in r]
        if len(xs) == 2 and len(ys) == 2:
            spread = max(max(xs) - min(xs), max(ys) - min(ys))
            diff = abs(sum(ys) / 2 - sum(xs) / 2)
            if diff < spread:
                print(f"  note: {label} difference ({diff:.4g}) is smaller than "
                      f"the within-arm spread ({spread:.4g}) - not a result")


if __name__ == "__main__":
    main()
