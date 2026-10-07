"""What runs between two verify graphs (torch profiler, one rank): every GPU
op from the last FlashMLA-step's final kernel to the next step's first
kernel, with gaps, plus the CPU-side runtime calls in that window.

    between_graphs.py <trace.json.gz> [--step N] [--all]
"""
import argparse
import collections
import gzip
import json
import statistics as st

MLA = "flash_fwd_splitkv_mla_fp8_sparse"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--step", type=int)
    a = ap.parse_args()
    ev = json.load(gzip.open(a.trace))["traceEvents"]
    gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                 key=lambda e: e["ts"])
    mla = [i for i, k in enumerate(gpu) if MLA in k["name"]]
    groups, cur = [], []
    for i in mla:
        if cur and gpu[i]["ts"] - gpu[cur[-1]]["ts"] > 1000:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    groups = [g for g in groups if len(g) == 78]
    # target graph end: the last kernel before a > 1 us run of non-graph work is
    # hard to tell; use the logits GEMM: the first kernel after the last
    # FlashMLA whose grid... simpler: the window starts at the last FlashMLA
    # and ends at the next step's first FlashMLA; report everything after the
    # last layer's MoE all-reduce.
    agg = collections.defaultdict(list)
    windows = []
    for g0, g1 in zip(groups, groups[1:]):
        lo, hi = g0[-1], g1[0]
        ks = gpu[lo:hi]
        # last-layer tail: skip until after the final allreduce_fusion of g0
        ar = [j for j, k in enumerate(ks) if "allreduce_fusion" in k["name"]
              or "cross_device_reduce" in k["name"]]
        start = ar[-1] + 1 if ar else 0
        # the next step's pre-attention of layer 0 begins with its embedding;
        # stop at the first kernel of the next verify graph: walk back from
        # the next FlashMLA to the last gap > 20 us
        end = len(ks)
        for j in range(len(ks) - 1, start, -1):
            if ks[j]["ts"] - (ks[j - 1]["ts"] + ks[j - 1]["dur"]) > 20:
                end = j
                break
        w = ks[start:end]
        windows.append((w, ks[end:]))
        t0 = w[0]["ts"]
        span = ks[end]["ts"] - t0 if end < len(ks) else 0
        busy = sum(k["dur"] for k in w)
        agg["span"].append(span)
        agg["busy"].append(busy)
        agg["n"].append(len(w))
    print(f"{len(windows)} inter-graph windows: span median {st.median(agg['span']):.0f} us, "
          f"GPU-busy sum {st.median(agg['busy']):.0f} us, kernels {st.median(agg['n']):.0f}")
    names = collections.defaultdict(list)
    for w, _ in windows:
        per = collections.defaultdict(float)
        for k in w:
            per[k["name"][:80]] += k["dur"]
        for n, v in per.items():
            names[n].append(v)
    print("top kernels in the window (median us per window, present-in share):")
    for n, v in sorted(names.items(), key=lambda x: -sum(x[1]))[:25]:
        print(f"  {sum(v) / len(windows):7.1f}  {len(v) / len(windows):4.0%}  {n}")
    w, nxt = windows[a.step if a.step is not None else len(windows) // 2]
    t0 = w[0]["ts"]
    print("\none window:")
    prev = None
    for k in w + nxt[:3]:
        gap = k["ts"] - (prev["ts"] + prev["dur"]) if prev else 0
        print(f"  {k['ts'] - t0:8.1f} gap {gap:6.1f} {k['dur']:7.1f}  s{k['args'].get('stream')}  {k['name'][:90]}")
        prev = k


main()
