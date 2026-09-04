import collections, statistics, sys
from pathlib import Path
SMEM = Path("agent_space/experiments/2026-07-29-marlin-smem-monopoly")
sys.path.insert(0, str(SMEM))
from analyze_step_budget import load_trace, step_windows, GPU_CATEGORIES, LAUNCH_CATEGORIES
root = Path("/e/project1/profound/alint77/traces/mtp3-profile-1665068")

def collect(label, needle, expect):
    per_rank = {}
    context = {}
    for path in sorted((root/label).glob("*.trace.json.gz")):
        rank = int(path.name.split("_rank")[1].split(".")[0])
        events = load_trace(path)
        bycorr = collections.defaultdict(list)
        for e in events:
            if e.get("cat") in GPU_CATEGORIES and e.get("args",{}).get("correlation") is not None:
                bycorr[e["args"]["correlation"]].append(e)
        launches = sorted((e for e in events if e.get("cat") in LAUNCH_CATEGORIES), key=lambda e: e["t"])
        steps = []
        for start, end in step_windows(events):
            corrs = [e["args"]["correlation"] for e in launches
                     if start <= e["t"] < end and "correlation" in e.get("args",{})]
            ops = sorted((k for c in corrs for k in bycorr[c]), key=lambda e: e["t"])
            idx = [i for i, e in enumerate(ops) if needle in e["name"]]
            if len(idx) != expect: continue
            steps.append([ops[i]["dur"] for i in idx])
            if rank == 0 and len(steps) == 1:
                for ordn, i in enumerate(idx[:14]):
                    prev = ops[i-1]["name"][:58] if i else "-"
                    nxt = ops[i+1]["name"][:58] if i+1 < len(ops) else "-"
                    context[ordn] = (prev, nxt)
        per_rank[rank] = steps
    return per_rank, context

for label, needle, expect, unit in (("decode","cross_device_reduce",166,"us"),
                                    ("prefill","ncclDevKernel_AllReduce",160,"us")):
    per_rank, context = collect(label, needle, expect)
    n = min(len(v) for v in per_rank.values())
    print(f"\n########## {label}: {n} aligned steps x {expect} ordinals, 4 ranks ##########")
    argmin = collections.Counter(); mins = []; alls = []
    allhigh = 0; total = 0
    for s in range(n):
        for o in range(expect):
            d = [per_rank[r][s][o] for r in sorted(per_rank)]
            mins.append(min(d)); alls.append(statistics.fmean(d))
            argmin[d.index(min(d))] += 1
            total += 1
            if min(d) > 3*mins[0] if False else False: pass
    floor = sorted(mins)[len(mins)//20]   # p5 of cross-rank minima
    for s in range(n):
        for o in range(expect):
            d = [per_rank[r][s][o] for r in sorted(per_rank)]
            if min(d) > 3*floor: allhigh += 1
    print(f"  cross-rank MIN per (step,ordinal): p50 {statistics.median(mins):8.1f} {unit}  "
          f"p5 {floor:8.1f}  mean {statistics.fmean(mins):8.1f}")
    print(f"  cross-rank MEAN per (step,ordinal): p50 {statistics.median(alls):8.1f} {unit}  "
          f"mean {statistics.fmean(alls):8.1f}")
    print(f"  => per-step: sum of minima {sum(mins)/n/1000:7.3f} ms vs "
          f"sum of means {sum(alls)/n/1000:7.3f} ms  "
          f"-> skew {sum(a-m for a,m in zip(alls,mins))/n/1000:7.3f} ms/step "
          f"({100*sum(a-m for a,m in zip(alls,mins))/sum(alls):.1f}%)")
    print(f"  ALL FOUR ranks slow (min > 3x p5-floor): {allhigh}/{total} = {100*allhigh/total:.1f}%"
          "   <- high => collective itself, low => arrival skew")
    print(f"  argmin rank histogram (who waits LEAST = arrives LAST): "
          + ", ".join(f"r{r} {100*c/total:5.1f}%" for r, c in sorted(argmin.items())))
    if context:
        print(f"  rank0 kernel context around first ordinals ({label}):")
        for o in sorted(context)[:12]:
            prev, nxt = context[o]
            print(f"    ord {o:2d} [{'even' if o%2==0 else 'odd '}] prev={prev}")
