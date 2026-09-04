import collections, statistics, sys
from pathlib import Path
SMEM = Path("agent_space/experiments/2026-07-29-marlin-smem-monopoly")
sys.path.insert(0, str(SMEM))
from analyze_step_budget import load_trace, step_windows, GPU_CATEGORIES, LAUNCH_CATEGORIES
root = Path("/e/project1/profound/alint77/traces/mtp3-profile-1665068")

per_rank_step_total = {}
ordinal_dur = collections.defaultdict(list)   # rank0 only: ordinal -> [dur per step]
for path in sorted((root / "decode").glob("*.trace.json.gz")):
    rank = int(path.name.split("_rank")[1].split(".")[0])
    events = load_trace(path)
    bycorr = collections.defaultdict(list)
    for e in events:
        if e.get("cat") in GPU_CATEGORIES and e.get("args", {}).get("correlation") is not None:
            bycorr[e["args"]["correlation"]].append(e)
    launches = sorted((e for e in events if e.get("cat") in LAUNCH_CATEGORIES), key=lambda e: e["t"])
    totals = []
    for wi, (start, end) in enumerate(step_windows(events)):
        corrs = [e["args"]["correlation"] for e in launches
                 if start <= e["t"] < end and "correlation" in e.get("args", {})]
        ops = sorted((k for c in corrs for k in bycorr[c]), key=lambda e: e["t"])
        ar = [e for e in ops if "cross_device_reduce" in e["name"]]
        if len(ar) != 166:
            continue
        totals.append(sum(e["dur"] for e in ar)/1000)
        if rank == 0:
            for i, e in enumerate(ar):
                ordinal_dur[i].append(e["dur"])
    per_rank_step_total[rank] = totals
    print(f"rank {rank}: {len(totals)} clean steps, all-reduce total/step "
          f"mean {statistics.fmean(totals):.3f} ms  median {statistics.median(totals):.3f}")

print("\nrank0: per-ordinal all-reduce duration across steps (top 20 by median)")
rows = sorted(((statistics.median(v), i, statistics.fmean(v), min(v), max(v), len(v))
               for i, v in ordinal_dur.items()), reverse=True)
for med, i, mean, lo, hi, n in rows[:20]:
    print(f"  ordinal {i:3d}: median {med:7.1f}us  mean {mean:7.1f}  min {lo:6.1f}  max {hi:7.1f}  n={n}")
big = [r for r in rows if r[0] > 50]
print(f"\nordinals with median > 50us: {len(big)}  -> {sorted(r[1] for r in big)}")
tot = sum(statistics.median(v) for v in ordinal_dur.values())
print(f"sum of per-ordinal medians {tot/1000:.3f} ms; "
      f"those {len(big)} ordinals contribute {sum(r[0] for r in big)/1000:.3f} ms "
      f"({100*sum(r[0] for r in big)/tot:.1f}%)")
