#!/usr/bin/env python3
"""Does the per-rank cold-expert burden explain the post-MoE all-reduce
stragglers? Joins this run's TD per-CTA dump (decode-cta/rank*-stop.pt, per
launch hot/cold counts, grouped into steps) with the torch trace's post-MoE
fused AR last-arrivals (minimum-duration rank per (step, ordinal)).

    moe_skew_probe.py <trace-dir> <td-dump-dir>
"""
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cta_analysis  # noqa: E402
import step_breakdown as S  # noqa: E402


def main():
    trace_dir, td_dir = Path(sys.argv[1]), Path(sys.argv[2])
    tables = {}
    for path in sorted(trace_dir.glob("*.pt.trace.json.gz")):
        rank = int(S.RANK_RE.search(path.name).group(1))
        rows = S.steps(S.load(path))
        tab = {}
        for index, row in enumerate(rows):
            prev = None
            ordinal = 0
            for op in sorted(row["phases"]["target"], key=lambda e: e["t"]):
                if "allreduce_fusion" in op["name"] and prev is not None \
                        and "nvjet" not in prev["name"]:
                    tab[(index, ordinal)] = op["end"] - op["t"]
                    ordinal += 1
                prev = op
        tables[rank] = tab
    n_steps = max(k[0] for k in tables[0]) + 1

    # per (trace step, rank): post-MoE ARs where the rank is last to arrive
    last = {}
    for index in range(n_steps):
        for o in sorted(k[1] for k in tables[0] if k[0] == index):
            vals = {rank: tables[rank][(index, o)] for rank in tables
                    if (index, o) in tables[rank]}
            if len(vals) < 4:
                continue
            floor = min(vals.values())
            winner = next(r for r, v in sorted(vals.items()) if v <= floor * 1.001)
            last[(index, winner)] = last.get((index, winner), 0) + 1

    nc, span = {}, {}
    for path in sorted(td_dir.glob("rank*-stop.pt")):
        rank = int(path.name[4])
        launches = cta_analysis.launches(torch.load(path))[0]  # w13, 132 CTAs each
        # group launches into decode steps: consecutive layers' MoE launches are
        # ~360 us apart, the next step's first MoE comes after the 2.7+ ms
        # lion-MLP / logits / drafter boundary
        steps, current = [], [launches[0]]
        for L in launches[1:]:
            if L[0]["t0"] - current[-1][0]["t0"] > 1_200_000:  # ns %globaltimer
                steps.append(current)
                current = []
            current.append(L)
        steps.append(current)
        steps = [s for s in steps if len(s) == 75]  # whole steps only
        for step, group in enumerate(steps):
            # each CTA row of a launch repeats the launch's nh/nc; take row 0
            nc[(step, rank)] = sum(L[0]["nc"] for L in group)
            span[(step, rank)] = sum(
                (max(r["t2"] for r in L) - min(r["t1"] for r in L if r["t1"])) / 1e3
                for L in group)
    n_td = max(s for s, _ in nc) + 1
    print(f"torch trace: {n_steps} steps; TD dump: {n_td} whole decode steps "
          f"(buffer holds the last steps before the dump)")

    for label, matrix in (("cold experts per step", nc),
                          ("w13 CTA time per step (us)", span)):
        pairs, tops, lasts = [], [], []
        for step in range(n_td):
            trace_step = n_steps - n_td + step
            pairs.append((step, trace_step))
            tops.append(max(range(4), key=lambda r: matrix.get((step, r), 0)))
            lasts.append(max(range(4), key=lambda r: last.get((trace_step, r), 0)))
        agree = sum(a == b for a, b in zip(lasts, tops))
        print(f"\n{label}:")
        print(f"  rank with the largest burden is also the AR last-arrival rank in "
              f"{agree}/{n_td} aligned steps: "
              + " ".join(f"td{s}/tr{t}:burden=r{c},last=r{l}"
                         for (s, t), c, l in zip(pairs, tops, lasts)))
        both = [(matrix.get((s, r), 0), last.get((t, r), 0))
                for s, t in pairs for r in range(4)]
        mx, my = statistics.fmean(x for x, _ in both), statistics.fmean(y for _, y in both)
        var = statistics.fmean((x - mx) ** 2 for x, _ in both)
        if var > 0:
            beta = statistics.fmean((x - mx) * (y - my) for x, y in both) / var
            print(f"  regression of AR-last-count on burden: {beta:+.3f} per unit "
                  f"(rank-mean burden {mx:.1f}, last-count {my:.1f})")


if __name__ == "__main__":
    main()
