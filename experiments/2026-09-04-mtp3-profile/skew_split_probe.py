import collections, statistics, sys
from pathlib import Path
SMEM = Path("agent_space/experiments/2026-07-29-marlin-smem-monopoly")
sys.path.insert(0, str(SMEM))
from analyze_step_budget import load_trace, step_windows, GPU_CATEGORIES, LAUNCH_CATEGORIES
root = Path("/e/project1/profound/alint77/traces/mtp3-profile-1665068")
path = sorted((root / "decode").glob("*_rank0.*trace.json.gz"))[0]
events = load_trace(path)
bycorr = collections.defaultdict(list)
for e in events:
    if e.get("cat") in GPU_CATEGORIES and e.get("args", {}).get("correlation") is not None:
        bycorr[e["args"]["correlation"]].append(e)
launches = sorted((e for e in events if e.get("cat") in LAUNCH_CATEGORIES), key=lambda e: e["t"])
od = collections.defaultdict(list)
n_steps = 0
for start, end in step_windows(events):
    corrs = [e["args"]["correlation"] for e in launches
             if start <= e["t"] < end and "correlation" in e.get("args", {})]
    ops = sorted((k for c in corrs for k in bycorr[c]), key=lambda e: e["t"])
    ar = [e for e in ops if "cross_device_reduce" in e["name"]]
    if len(ar) != 166: continue
    n_steps += 1
    for i, e in enumerate(ar): od[i].append(e["dur"])
ev = [i for i in od if i % 2 == 0]; odd = [i for i in od if i % 2 == 1]
def agg(idx):
    meds = [statistics.median(od[i]) for i in idx]
    return len(idx), statistics.fmean(meds), sum(meds)/1000
for name, idx in (("even ordinals", ev), ("odd ordinals", odd)):
    n, m, s = agg(idx)
    print(f"{name}: n={n}  mean-of-medians {m:7.1f} us  sum {s:6.3f} ms/step")
floor = statistics.fmean([min(od[i]) for i in od])
print(f"\nfloor (mean of per-ordinal MIN over {n_steps} steps): {floor:.2f} us")
print(f"166 x floor = {166*floor/1000:.3f} ms/step  <- hardware cost of the collective")
tot = sum(statistics.median(od[i]) for i in od)/1000
print(f"sum of per-ordinal medians = {tot:.3f} ms/step")
print(f"=> wait absorbed at the barrier = {tot - 166*floor/1000:.3f} ms/step")
