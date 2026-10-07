"""Cross-rank collective waiting in decode steps (torch profiler, all ranks).

Steps are the 78-FlashMLA groups; within a step every collective kernel gets
an ordinal per kind. For each (step, kind, ordinal) the minimum duration over
the ranks approximates the transfer; each rank's excess is waiting for the
last rank to arrive. Kinds: attention all-reduce (+RMSNorm) and MoE all-reduce
(the AR fusion kernel alternates attn/MoE within a layer), DCP query
gather_cat, LSE reduce-scatter, indexer gather.

    coll_wait.py <window dir>
"""
import collections
import gzip
import json
import statistics as st
import sys
from pathlib import Path

MLA = "flash_fwd_splitkv_mla_fp8_sparse"
KINDS = (("ar", "allreduce_fusion"), ("ar1", "cross_device_reduce"),
         ("gather_cat", "gather_cat_kernel"), ("lse_rs", "lse_reduce_scatter"),
         ("ix_gather", "one_shot::gather_kernel"))


def steps(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    kern = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    mla = [i for i, k in enumerate(kern) if MLA in k["name"]]
    groups, cur = [], []
    for i in mla:
        if cur and kern[i]["ts"] - kern[cur[-1]]["ts"] > 1000:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    out = []
    for g in groups:
        if len(g) != 78:
            continue
        # from the previous step's last FlashMLA tail to this step's last one
        lo = kern[g[0]]["ts"] - 400
        hi = kern[g[-1]]["ts"] + 400
        ks = [k for k in kern if lo <= k["ts"] < hi]
        row = {"t0": kern[g[0]]["ts"]}
        for kind, pat in KINDS:
            row[kind] = [k["dur"] for k in ks if pat in k["name"]]
        # attn vs MoE AR: alternate by position relative to FlashMLA calls
        mts = [kern[i]["ts"] for i in g]
        attn, moe = [], []
        import bisect
        for k in ks:
            if "allreduce_fusion" in k["name"]:
                j = bisect.bisect_right(mts, k["ts"]) - 1
                # within a layer: first AR after FlashMLA is attention's
                (attn if j >= 0 and not any(
                    mts[j] < x < k["ts"] for x in row.get("_seen", [])) else moe).append(k)
                row.setdefault("_seen", []).append(k["ts"])
        row["ar_attn"] = [k["dur"] for k in attn]
        row["ar_moe"] = [k["dur"] for k in moe]
        out.append(row)
    return out


def main():
    files = sorted(Path(sys.argv[1]).glob("*rank*.pt.trace.json.gz"))
    ranks = [steps(f) for f in files]
    # align steps across ranks by start time
    base = ranks[0]
    aligned = []
    for s in base:
        row = [s]
        for other in ranks[1:]:
            m = min(other, key=lambda o: abs(o["t0"] - s["t0"]))
            if abs(m["t0"] - s["t0"]) > 3000:
                break
            row.append(m)
        if len(row) == len(ranks):
            aligned.append(row)
    print(f"{len(aligned)} aligned steps x {len(ranks)} ranks")
    for kind in ("ar_attn", "ar_moe", "ar1", "gather_cat", "lse_rs", "ix_gather"):
        tot = collections.defaultdict(float)
        mins, n = 0.0, None
        ok = 0
        for row in aligned:
            lens = {len(r[kind]) for r in row}
            if len(lens) != 1:
                continue
            ok += 1
            n = lens.pop()
            for o in range(n):
                d = [r[kind][o] for r in row]
                mn = min(d)
                mins += mn
                for ri, x in enumerate(d):
                    tot[ri] += x
        if not ok:
            continue
        per = {ri: tot[ri] / ok / 1e3 for ri in tot}
        print(f"{kind:11s} x{n:3d}/step  transfer(min) {mins / ok / 1e3:.3f} ms/step  "
              f"rank total " + " ".join(f"{per[ri]:.3f}" for ri in sorted(per))
              + f"  -> wait mean {st.mean(per.values()) - mins / ok / 1e3:.3f}")


main()
