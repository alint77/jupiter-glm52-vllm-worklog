"""Activation peak from a snapshot's allocator trace: replay alloc/free
events, find the moment of maximum live bytes, and itemise the blocks live
then that were allocated after `--since` (default: the first event in the
ring), grouped by allocating vLLM code.

    mem_peak.py <snap.pickle> [--depth 3] [--top 30]
"""
import argparse
import collections
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_snapshot import key  # noqa: E402

G = 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    s = pickle.load(open(a.snap, "rb"))
    tr = s["device_traces"][0]
    live, cur, peak, peak_i = {}, 0, 0, 0
    for i, e in enumerate(tr):
        if e["action"] == "alloc":
            live[e["addr"]] = (e["size"], i)
            cur += e["size"]
            if cur > peak:
                peak, peak_i = cur, i
        elif e["action"] == "free_completed" and e["addr"] in live:
            cur -= live.pop(e["addr"])[0]
    # rebuild the live set at the peak
    live, cur = {}, 0
    for i, e in enumerate(tr[:peak_i + 1]):
        if e["action"] == "alloc":
            live[e["addr"]] = e
        elif e["action"] == "free_completed":
            live.pop(e["addr"], None)
    t0 = tr[0]["time_us"]
    print(f"{len(tr)} events over {(tr[-1]['time_us'] - t0) / 1e6:.1f} s; peak of trace-allocated "
          f"live bytes {peak / G:.3f} GiB at event {peak_i} (t+{(tr[peak_i]['time_us'] - t0) / 1e6:.1f} s)")
    by = collections.defaultdict(lambda: [0, 0])
    for e in live.values():
        k = key(e.get("frames", []), a.depth)
        by[k][0] += e["size"]
        by[k][1] += 1
    for k, (b, n) in sorted(by.items(), key=lambda x: -x[1][0])[:a.top]:
        print(f"{b / G:8.3f} GiB {n:5d}  {k}")


main()
