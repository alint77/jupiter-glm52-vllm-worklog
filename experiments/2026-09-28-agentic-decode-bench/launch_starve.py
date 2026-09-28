#!/usr/bin/env python3
"""Is a phase of the decode step host-launch starved? Rank 0, per profiler window.

For each step (analyze.py's step and phase split), walk the phase's GPU kernels
in start order. The idle gap before kernel i counts as *starved* when its CPU
launch call (cudaLaunchKernel / cuLaunchKernel / graph launch, matched by
correlation id) had not yet returned when kernel i-1 finished: the GPU ran out
of work because the host had not issued the next kernel. A captured graph
issues all its kernels at once, so the target's verify graph is the control.

    launch_starve.py <window dir>...
"""

import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402


def main() -> None:
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in map(Path, sys.argv[1:]):
        f = sorted(d.glob("*rank0*.pt.trace.json.gz"))[0]
        events = A.load(f)
        launch_end = {e["args"].get("correlation"): e["end"] for e in events
                      if e.get("cat") in A.LAUNCH_CATS}
        for row in A.steps(events):
            whole = [o for ops in row["phases"].values() for o in ops]
            for phase, ops in (*row["phases"].items(), ("whole step", whole)):
                ops = sorted(ops, key=lambda o: o["t"])
                if not ops:
                    continue
                span = ops[-1]["end"] - ops[0]["t"]
                busy = A.union(ops)
                starved = gaps = 0.0
                prev_end = ops[0]["end"]
                for o in ops[1:]:
                    gap = o["t"] - prev_end
                    if gap > 0:
                        gaps += gap
                        issued = launch_end.get(o["args"].get("correlation"))
                        if issued is not None and issued > prev_end:
                            starved += min(gap, issued - prev_end)
                    prev_end = max(prev_end, o["end"])
                p = per[phase]
                p["kernels"].append(len(ops))
                p["span"].append(span)
                p["busy"].append(busy)
                p["gaps"].append(gaps)
                p["starved"].append(starved)
                p["period"].append(row.get("period", 0.0))
    m = statistics.fmean
    steps = len(next(iter(per.values()))["span"])
    print(f"{steps} steps, rank 0; ms per step")
    print(f"{'phase':10s} {'kernels':>8s} {'span':>7s} {'GPU busy':>9s} {'idle':>7s} "
          f"{'starved':>8s}")
    for phase, p in per.items():
        print(f"{phase:10s} {m(p['kernels']):8.0f} {m(p['span']):7.2f} {m(p['busy']):9.2f} "
              f"{m(p['gaps']):7.2f} {m(p['starved']):8.2f}")
    print(f"step period {m(per['target']['period']):.2f} ms")


if __name__ == "__main__":
    main()
