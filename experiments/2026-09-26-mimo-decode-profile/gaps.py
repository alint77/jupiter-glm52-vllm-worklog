#!/usr/bin/env python3
"""The GPU's idle time in a decode step, and what runs around each gap.

    gaps.py <trace-dir> [--rank 0] [--step N]

Per step: every GPU op outside the verify graph in time order, with the idle
gap before it; then the median idle per gap position over all steps (the op
sequence outside the graphs repeats, so gaps line up by position).
"""

import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import RANK_RE, load, steps  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()
    path = next(p for p in sorted(args.trace_dir.glob("*.pt.trace.json.gz"))
                if int(RANK_RE.search(p.name).group(1)) == args.rank)
    rows = steps(load(path))
    per_pos = collections.defaultdict(list)
    names = {}
    for s, row in enumerate(rows):
        seq = []
        for phase, ops in row["phases"].items():
            if phase == "target":
                seq.append((min(o["t"] for o in ops), max(o["end"] for o in ops), "[verify graph]", phase))
            elif phase == "draft":
                # the drafter's graphs, as one block per graph launch is too fine; keep ops
                seq += [(o["t"], o["end"], o["name"], phase) for o in ops]
            else:
                seq += [(o["t"], o["end"], o["name"], phase) for o in ops]
        seq.sort()
        t0 = row["span"][0]
        cursor = t0
        out = []
        for i, (a, b, name, phase) in enumerate(seq):
            gap = max(0.0, a - cursor)
            out.append((gap, a, b, name, phase))
            cursor = max(cursor, b)
        end_gap = t0 + row["period"] - cursor
        if s == args.step:
            print(f"step {s}: period {row['period'] * 1000:.0f} us")
            for gap, a, b, name, phase in out:
                if gap * 1000 > 3 or name == "[verify graph]" or b - a > 0.02:
                    print(f"  +{(a - t0) * 1000:8.1f}  gap {gap * 1000:6.1f}  dur {(b - a) * 1000:7.1f}  {phase:9s} {name[:90]}")
            print(f"  end gap {end_gap * 1000:.1f}")
        key = tuple(n[:40] for _, _, _, n, _ in out)
        for i, (gap, _, _, name, phase) in enumerate(out):
            per_pos[(len(key), i)].append(gap * 1000)
            names[(len(key), i)] = (phase, name[:70])
        per_pos[(len(key), "end")].append(end_gap * 1000)
        names[(len(key), "end")] = ("", "end of step")
    common = collections.Counter(k[0] for k in per_pos).most_common(1)[0][0]
    total = 0.0
    print(f"\nmedian gap before each op (steps with the common {common}-op sequence):")
    for key, gaps in per_pos.items():
        if key[0] != common:
            continue
        med = statistics.median(gaps)
        total += statistics.mean(gaps)
        if med > 2:
            print(f"  {med:7.1f} us  before {names[key][0]:9s} {names[key][1]}")
    print(f"mean idle per step: {total:.0f} us")


if __name__ == "__main__":
    main()
