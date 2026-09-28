#!/usr/bin/env python3
"""Two consecutive decode steps from one profiler window, as Gantt data.

Picks the pair of steps on rank 0 whose periods sit closest to the window's
median, then cuts every rank's GPU kernels to that same time span. Each kernel
gets analyze.py's category (target) or its phase (draft / host-side / logits),
folded into the write-up's eight buckets. Rank 0 also gets its CPU launch calls
and its GPU idle gaps, each marked host-starved when the next kernel's launch
call had not returned when the GPU went idle (launch_starve.py's rule).

    gantt_data.py <window dir> <out.json>
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402

BUCKETS = ["MoE", "dense GEMM", "attention + indexer", "TP all-reduce",
           "DCP collectives", "drafter", "small unfused kernels", "host-side & logits"]
TARGET = {
    "MoE one-kernel w13": 0, "MoE one-kernel w2": 0,
    "MoE one-kernel route/act/finalize": 0, "MoE routing/align/sum/act": 0,
    "MoE hot Marlin (HBM)": 0, "MoE cold Marlin (Grace)": 0,
    "dense GEMM": 1, "attention": 2, "DSA indexer": 2, "TP all-reduce": 3,
    "DCP one-shot collectives": 4, "norm/rope/elementwise": 6,
}


def bucket(phase: str, op: dict) -> int:
    if phase == "draft":
        return 5
    if phase != "target":
        return 7
    if op["name"].startswith("triton_tem_fused_mm"):
        return 1
    return TARGET.get(A.category(op), 6)


def main() -> None:
    window, out = Path(sys.argv[1]), Path(sys.argv[2])
    files = {int(A.RANK_RE.search(p.name).group(1)): p
             for p in window.glob("*.pt.trace.json.gz")}
    events0 = A.load(files[0])
    rows0 = A.steps(events0)
    med = statistics.median(r["period"] for r in rows0)
    i = min(range(len(rows0) - 1),
            key=lambda k: abs(rows0[k]["period"] - med) + abs(rows0[k + 1]["period"] - med))
    t0 = rows0[i]["span"][0]
    t1 = rows0[i + 1]["span"][0] + rows0[i + 1]["period"]
    names: dict[str, int] = {}
    data = {"buckets": BUCKETS, "t0": 0.0, "t1": round((t1 - t0) * 1000, 1),
            "steps": [], "ranks": {}, "names": []}
    for k in (i, i + 1):
        data["steps"].append({"start": round((rows0[k]["span"][0] - t0) * 1000, 1),
                              "period_ms": round(rows0[k]["period"], 3)})

    def name_id(n: str) -> int:
        n = n[:140]
        if n not in names:
            names[n] = len(names)
        return names[n]

    for rank, path in sorted(files.items()):
        events = events0 if rank == 0 else A.load(path)
        rows = rows0 if rank == 0 else A.steps(events)
        kernels = []
        for row in rows:
            for phase, ops in row["phases"].items():
                for o in ops:
                    if o["end"] < t0 or o["t"] > t1:
                        continue
                    kernels.append([round((o["t"] - t0) * 1000, 2),
                                    round((o["end"] - o["t"]) * 1000, 2),
                                    bucket(phase, o), name_id(o["name"]),
                                    str(o.get("tid")), phase])
        kernels.sort()
        entry = {"kernels": kernels}
        if rank == 0:
            launches = {}
            host = []
            for e in events:
                if e.get("cat") in A.LAUNCH_CATS and t0 <= e["t"] <= t1:
                    launches[e["args"].get("correlation")] = e["end"]
                    host.append([round((e["t"] - t0) * 1000, 2),
                                 round((e["end"] - e["t"]) * 1000, 2),
                                 1 if "Graph" in e["name"] else 0])
            gaps = []
            corr = {}
            for row in rows:
                for ops in row["phases"].values():
                    for o in ops:
                        corr[(round(o["t"], 6), o["name"])] = o["args"].get("correlation")
            ordered = sorted((o for row in rows for ops in row["phases"].values()
                              for o in ops if t0 <= o["t"] <= t1), key=lambda o: o["t"])
            prev_end = ordered[0]["end"]
            for o in ordered[1:]:
                gap = o["t"] - prev_end
                if gap > 0:
                    issued = launches.get(o["args"].get("correlation"))
                    starved = issued is not None and issued > prev_end
                    gaps.append([round((prev_end - t0) * 1000, 2), round(gap * 1000, 2),
                                 1 if starved else 0])
                prev_end = max(prev_end, o["end"])
            entry["host"] = host
            entry["gaps"] = gaps
        data["ranks"][rank] = entry
    data["names"] = [n for n, _ in sorted(names.items(), key=lambda kv: kv[1])]
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"steps {i},{i + 1}: periods {[s['period_ms'] for s in data['steps']]} ms; "
          f"span {data['t1'] / 1000:.2f} ms; kernels "
          f"{[len(r['kernels']) for r in data['ranks'].values()]}; "
          f"{len(data['names'])} names; {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
