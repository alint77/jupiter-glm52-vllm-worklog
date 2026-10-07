"""Where the MoE all-reduce wait comes from (torch profiler, all ranks).

Per (step, layer) and rank, between the attention all-reduce end (a sync
point: every rank leaves it together) and the MoE all-reduce start (arrival):
the MoE chain span (route_prep start -> finalize end) and the rest. The MoE
AR wait of the last-arriving rank is ~0, so the spread of arrivals is the
wait; this splits it into the MoE-chain spread and the non-MoE spread.

    moe_skew.py <window dir>
"""
import bisect
import gzip
import json
import statistics as st
import sys
from pathlib import Path

MLA = "flash_fwd_splitkv_mla_fp8_sparse"


def layers(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    kern = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    mla = [i for i, k in enumerate(kern) if MLA in k["name"]]
    steps, cur = [], []
    for i in mla:
        if cur and kern[i]["ts"] - kern[cur[-1]]["ts"] > 1000:
            steps.append(cur)
            cur = []
        cur.append(i)
    steps.append(cur)
    out = []
    for g in steps:
        if len(g) != 78:
            continue
        rows = []
        for L in range(3, 77):  # MoE layers with a following FlashMLA
            ks = kern[g[L]:g[L + 1]]
            ar = [k for k in ks if "allreduce_fusion" in k["name"]]
            prep = [k for k in ks if "route_prep" in k["name"]]
            fin = [k for k in ks if "finalize_kernel" in k["name"]]
            if len(ar) < 2 or not prep or not fin:
                rows.append(None)
                continue
            a0 = ar[0]["ts"] + ar[0]["dur"]
            rows.append({"chain": fin[0]["ts"] + fin[0]["dur"] - prep[0]["ts"],
                         "arrive": ar[1]["ts"] - a0, "pre": prep[0]["ts"] - a0,
                         "wait": ar[1]["dur"]})
        out.append((kern[g[0]]["ts"], rows))
    return out


def main():
    files = sorted(Path(sys.argv[1]).glob("*rank*.pt.trace.json.gz"))
    R = [layers(f) for f in files]
    spread_arrive, spread_chain, spread_pre, slow_is_chain = [], [], [], 0
    n = 0
    for t0, rows in R[0]:
        others = []
        for r in R[1:]:
            m = min(r, key=lambda s: abs(s[0] - t0))
            if abs(m[0] - t0) > 3000:
                break
            others.append(m[1])
        if len(others) != len(R) - 1:
            continue
        for L in range(len(rows)):
            per = [rows[L]] + [o[L] for o in others]
            if any(p is None for p in per):
                continue
            n += 1
            arr = [p["arrive"] for p in per]
            ch = [p["chain"] for p in per]
            pre = [p["pre"] for p in per]
            spread_arrive.append(max(arr) - st.mean(arr))
            spread_chain.append(max(ch) - st.mean(ch))
            spread_pre.append(max(pre) - st.mean(pre))
            slow = max(range(len(per)), key=lambda i: arr[i])
            slow_is_chain += ch[slow] == max(ch)
    print(f"{n} (step, layer) cases; per-layer mean over cases, us:")
    print(f"  arrival spread (max - mean)   {st.mean(spread_arrive):6.1f}  -> x74 layers "
          f"= {st.mean(spread_arrive) * 74 / 1e3:.2f} ms/step mean wait")
    print(f"  MoE chain spread (max - mean) {st.mean(spread_chain):6.1f}")
    print(f"  pre-chain spread (max - mean) {st.mean(spread_pre):6.1f}")
    print(f"  slowest-arriving rank also has the longest MoE chain in {slow_is_chain / n:.0%}")
    q = sorted(spread_chain)
    print(f"  chain spread p50 {q[len(q) // 2]:.1f} p90 {q[int(.9 * len(q))]:.1f} max {q[-1]:.1f}")


main()
