"""One decode step's per-layer kernel timeline (torch profiler, one rank).

Layers are delimited by the FlashMLA sparse decode call (one per layer). For
the chosen layers prints every kernel between this layer's FlashMLA and the
next one, with start offset from the FlashMLA start, duration and stream.
Also prints, per layer index over all steps, the median layer period.

    layer_timeline.py <trace.json.gz> [--step N] [--layers 3,40,41,42]
"""
import argparse
import gzip
import json
import statistics as st

MLA = "flash_fwd_splitkv_mla_fp8_sparse"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--layers", default="3,40,41,42")
    a = ap.parse_args()
    ev = json.load(gzip.open(a.trace))["traceEvents"]
    kern = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                  key=lambda e: e["ts"])
    mla = [k for k in kern if MLA in k["name"]]
    steps, cur = [], []
    for k in mla:
        if cur and k["ts"] - cur[-1]["ts"] > 1000:
            steps.append(cur)
            cur = []
        cur.append(k)
    steps.append(cur)
    steps = [s for s in steps if len(s) == 78]
    per = [[s[i + 1]["ts"] - s[i]["ts"] for s in steps] for i in range(77)]
    print(f"{len(steps)} steps; median layer period (MLA_i -> MLA_i+1), us:")
    print(" ".join(f"{i}:{st.median(p):.0f}" for i, p in enumerate(per)))
    s = steps[a.step if a.step is not None else len(steps) // 2]
    starts = [k["ts"] for k in kern]
    import bisect
    for L in map(int, a.layers.split(",")):
        t0 = s[L]["ts"]
        t1 = s[L + 1]["ts"] if L + 1 < 78 else t0 + 400
        print(f"\n== layer {L}: period {t1 - t0:.1f} us")
        i = bisect.bisect_left(starts, t0 - 1)
        while i < len(kern) and kern[i]["ts"] < t1:
            k = kern[i]
            print(f"  {k['ts'] - t0:8.1f} {k['dur']:7.1f}  s{k['args'].get('stream')}"
                  f"  g{k['args'].get('grid', ['?'])}  {k['name'][:90]}")
            i += 1


main()
