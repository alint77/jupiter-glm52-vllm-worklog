#!/usr/bin/env python3
"""Map each GPU kernel of an eager (no CUDA graph) stack-traced torch profile
back to the vLLM source line that launched it.

For every kernel in the decode steps (analyze.py's step split): the launch
call (cudaLaunchKernel & co, by correlation id) sits inside a stack of
python_function events on the same thread; the innermost frame under vllm/
is the attribution. Reports per (kernel, vllm frame): launches and GPU time
per step, largest first, optionally only kernels under --max-us each.

    attribute_kernels.py <window dir> [--rank 0] [--max-us 8] [--top 60]
"""

import argparse
import bisect
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("window", type=Path)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--max-us", type=float, default=1e9)
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()
    f = sorted(args.window.glob(f"*rank{args.rank}.*.pt.trace.json.gz"))[0]
    events = A.load(f)
    launch = {e["args"].get("correlation"): e for e in events if e.get("cat") in A.LAUNCH_CATS}
    py = collections.defaultdict(list)
    for e in events:
        if e.get("cat") == "python_function":
            py[e["tid"]].append(e)
    starts = {}
    for tid in py:
        py[tid].sort(key=lambda e: e["t"])
        starts[tid] = [e["t"] for e in py[tid]]

    def frame(l):
        tid = l["tid"]
        if tid not in py:
            return "?"
        i = bisect.bisect_right(starts[tid], l["t"])
        best = None
        # enclosing frames: scan back a bounded window
        for e in reversed(py[tid][max(0, i - 4000):i]):
            if e["end"] >= l["end"] and "vllm/" in e["name"] and ".venv" not in e["name"]:
                if best is None or e["t"] > best["t"]:
                    best = e
        if best is None:
            return "?"
        name = best["name"]
        return name[name.find("vllm/"):][:110]

    windows = sorted((e for e in events if e.get("cat") == "user_annotation"
                      and e["name"].startswith("execute_")), key=lambda e: e["t"])
    wstart = [w["t"] for w in windows]
    by_corr = collections.defaultdict(list)
    for e in events:
        if e.get("cat") in A.GPU_CATS:
            by_corr[e["args"].get("correlation")].append(e)
    nsteps = max(1, len(windows) - 1)
    cnt = collections.Counter()
    dur = collections.defaultdict(float)
    for corr, l in launch.items():
        i = bisect.bisect_right(wstart, l["t"]) - 1
        if not 0 <= i < len(windows) - 1:
            continue
        for o in by_corr.get(corr, []):
            d = o["end"] - o["t"]
            if d * 1000 > args.max_us:
                continue
            k = (o["name"][:60], frame(l))
            cnt[k] += 1
            dur[k] += d
    print(f"{nsteps} steps, rank {args.rank}; kernels <= {args.max_us} us; per step:")
    for k in sorted(dur, key=lambda k: -cnt[k])[: args.top]:
        print(f"x{cnt[k] / nsteps:6.1f} {dur[k] / nsteps * 1000:8.1f} us  {k[0]:60s} <- {k[1]}")


if __name__ == "__main__":
    main()
