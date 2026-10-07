"""Everything on the GPU between step N's last FlashMLA and step N+1's first
(torch profiler, one rank), grouped into runs by the CPU call that launched
them (cudaGraphLaunch id or eager launches), with idle gaps.

    step_tail.py <trace.json.gz> [--step N]
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
    launch = {e["args"].get("correlation"): e for e in ev
              if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    mla = [i for i, k in enumerate(gpu) if MLA in k["name"]]
    groups, cur = [], []
    for i in mla:
        if cur and gpu[i]["ts"] - gpu[cur[-1]]["ts"] > 1000:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    groups = [g for g in groups if len(g) == 78]
    pairs = list(zip(groups, groups[1:]))
    tot = collections.defaultdict(list)
    for g0, g1 in pairs:
        ks = gpu[g0[-1]:g1[0] + 1]
        span = ks[-1]["ts"] - ks[0]["ts"]
        runs = []
        for k in ks:
            c = launch.get(k["args"].get("correlation"))
            key = (c["name"], c["ts"]) if c and "Graph" in c["name"] else ("eager", 0)
            if runs and runs[-1][0] == key:
                runs[-1][1].append(k)
            else:
                runs.append((key, [k]))
        idle = 0.0
        end = ks[0]["ts"]
        for k in ks:
            if k["ts"] > end:
                idle += k["ts"] - end
            end = max(end, k["ts"] + k["dur"])
        tot["span"].append(span)
        tot["idle"].append(idle)
        tot["runs"].append(runs)
    print(f"{len(pairs)} steps: last->first FlashMLA span median {st.median(tot['span']):.0f} us, "
          f"GPU idle in it {st.median(tot['idle']):.0f} us")
    runs = tot["runs"][a.step if a.step is not None else len(pairs) // 2]
    t0 = runs[0][1][0]["ts"]
    prev_end = t0
    for (kind, _), ks in runs:
        s, e = ks[0]["ts"], max(k["ts"] + k["dur"] for k in ks)
        busy = sum(k["dur"] for k in ks)
        top = collections.Counter(k["name"][:50] for k in ks).most_common(3)
        print(f"  {s - t0:8.1f}  gap {s - prev_end:6.1f}  span {e - s:7.1f}  busy {busy:7.1f}  "
              f"{kind:24s} n={len(ks):4d}  {top}")
        prev_end = max(prev_end, e)


main()
