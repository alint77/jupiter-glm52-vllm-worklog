"""Per-step kernel time by name inside the verify graph (breakdown2.py's
step segmentation), for one category key or all: what moved between windows.

    kernels.py <trace.json.gz> [--match substr ...] [--top 25]
"""
import argparse
import bisect
import collections
import gzip
import json

from breakdown2 import CATS, GPU, LAUNCH, cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--cat", type=int, nargs="*")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--outside", action="store_true", help="kernels outside the verify graph")
    a = ap.parse_args()
    ev = json.load(gzip.open(a.trace))["traceEvents"]
    by_corr = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in GPU:
            by_corr[e.get("args", {}).get("correlation")].append(e)
    starts = sorted(e["ts"] for e in ev if e.get("cat") == "user_annotation"
                    and e["name"].startswith("execute_"))
    per_step = collections.defaultdict(list)
    for la in (e for e in ev if e.get("cat") in LAUNCH):
        i = bisect.bisect(starts, la["ts"]) - 1
        if 0 <= i < len(starts) - 1:
            per_step[i].append(la)
    tot = collections.Counter()
    cnt = collections.Counter()
    n = 0
    for i, las in per_step.items():
        graphs = [la for la in las if "GraphLaunch" in la["name"]]
        if not graphs:
            continue
        t = max(graphs, key=lambda la: len(by_corr[la["args"]["correlation"]]))
        ks = by_corr[t["args"]["correlation"]]
        if len(ks) < 500:
            continue
        n += 1
        if a.outside:
            tc = t["args"]["correlation"]
            ks = [k for la in las if la["args"].get("correlation") != tc
                  for k in by_corr.get(la["args"].get("correlation"), [])]
        for k in ks:
            c = cat(k["name"])
            if a.outside or a.cat is None or c in a.cat:
                key = (CATS[c][0][:14], k["name"][:110])
                tot[key] += k["dur"]
                cnt[key] += 1
    print(f"{n} verify graphs")
    for (c, name), us in tot.most_common(a.top):
        print(f"{us / n / 1e3:7.3f} ms {cnt[(c, name)] / n:6.0f}/step {us / cnt[(c, name)]:7.1f} us  {c:14s} {name}")


if __name__ == "__main__":
    main()
