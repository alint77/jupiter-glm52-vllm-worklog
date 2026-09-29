#!/usr/bin/env python3
"""Who is late into the verify graph's first all-reduce, and why?

Per step and rank (analyze.py's step split): the CPU time of the target graph
launch, the GPU start of its first kernel, and the first cross_device_reduce's
duration (its wait). Steps are matched across ranks by that all-reduce's end,
which completes on all ranks together. For the rank that entered last, the
lateness is split into host (the graph launch call itself came late) and GPU
(the launch was issued in time but the stream was still busy).

    entry_stall.py <window dir>...
"""

import bisect
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402


def rank_rows(path):
    events = A.load(path)
    launch = {e["args"].get("correlation"): e for e in events if e.get("cat") in A.LAUNCH_CATS}
    cpu = sorted((e for e in events if e.get("cat") in ("python_function", "cpu_op",
                                                        "user_annotation")),
                 key=lambda e: e["t"])
    rows = []
    for row in A.steps(events):
        target = sorted(row["phases"]["target"], key=lambda o: o["t"])
        ar = [o for o in target if "cross_device_reduce" in o["name"]]
        if not target or not ar:
            continue
        gl = launch.get(target[0]["args"].get("correlation"))
        prev_gpu_end = max((o["end"] for ph, ops in row["phases"].items() if ph != "target"
                            for o in ops if o["t"] < target[0]["t"]), default=None)
        draft = sorted(row["phases"].get("draft", []), key=lambda o: o["t"])
        rows.append({
            "entry": target[0]["t"], "launch": gl["t"] if gl else None,
            "ar0": ar[0]["end"] - ar[0]["t"], "ar0_end": ar[0]["end"],
            "prev_gpu_end": prev_gpu_end, "period": row.get("period"),
            "draft_span": (draft[-1]["end"] - draft[0]["t"]) if draft else 0.0,
        })
    return rows, cpu


def main():
    late_by = collections.Counter()
    records = []
    for d in map(Path, sys.argv[1:]):
        ranks = {}
        for f in sorted(d.glob("*rank*.pt.trace.json.gz")):
            r = int(f.name.split("_rank")[1].split(".")[0])
            ranks[r] = rank_rows(f)
        base = ranks[0][0]
        ends = {r: [x["ar0_end"] for x in rows] for r, (rows, _) in ranks.items()}
        for i, row0 in enumerate(base):
            group = {0: row0}
            for r in ranks:
                if r == 0:
                    continue
                j = bisect.bisect(ends[r], row0["ar0_end"])
                cands = [k for k in (j - 1, j) if 0 <= k < len(ends[r])]
                k = min(cands, key=lambda k: abs(ends[r][k] - row0["ar0_end"]))
                if abs(ends[r][k] - row0["ar0_end"]) < 0.05:
                    group[r] = ranks[r][0][k]
            if len(group) < len(ranks):
                continue
            wait = max(g["ar0"] for g in group.values())
            last = max(group, key=lambda r: group[r]["entry"])
            first_entry = min(g["entry"] for g in group.values())
            g = group[last]
            lateness = g["entry"] - first_entry
            # launch issued after the other ranks were already running?
            host = max(0.0, (g["launch"] or g["entry"]) - first_entry) if g["launch"] else 0.0
            records.append((d.name, i, wait, last, lateness, min(host, lateness), g, group))
    waits = [r[2] for r in records]
    print(f"{len(records)} matched steps; max-rank AR#0 wait mean {statistics.fmean(waits):.3f} "
          f"p50 {statistics.median(waits):.3f} ms")
    big = [r for r in records if r[2] > 0.5]
    print(f"steps with wait > 0.5 ms: {len(big)} ({sum(r[2] for r in big) / len(records):.3f} ms/step)")
    for r in big:
        late_by[r[3]] += 1
    print("last rank to enter in those steps:", dict(late_by))
    host_share = sum(r[5] for r in big) / max(1e-9, sum(r[4] for r in big))
    print(f"lateness explained by a late graph-launch call (host): {host_share:.0%}")
    print("\nwin step  wait  last  late  host  last-rank: launch->entry  prev-gpu-end->launch  "
          "draft span  period")
    for w, i, wait, last, late, host, g, _ in sorted(big, key=lambda r: -r[2])[:25]:
        le = (g["entry"] - g["launch"]) if g["launch"] else float("nan")
        pg = (g["launch"] - g["prev_gpu_end"]) if g["launch"] and g["prev_gpu_end"] else float("nan")
        print(f"{w[-1]:>3s} {i:4d} {wait:5.2f}  r{last}  {late:5.2f} {host:5.2f}   "
              f"{le:8.3f}             {pg:8.3f}            {g['draft_span']:6.2f}  "
              f"{(g['period'] or 0):6.2f}")


if __name__ == "__main__":
    main()
