import collections, statistics, sys
from pathlib import Path
SMEM = Path("agent_space/experiments/2026-07-29-marlin-smem-monopoly")
sys.path.insert(0, str(SMEM))
from analyze_step_budget import load_trace, step_windows, GPU_CATEGORIES, LAUNCH_CATEGORIES
root = Path("/e/project1/profound/alint77/traces/mtp3-profile-1665068")
def union(evts):
    iv = sorted((e["t"], e["t"]+e["dur"]/1000) for e in evts)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None: tot += ce-cs
            cs, ce = s, e
        else: ce = max(ce, e)
    return tot + (ce-cs if ce is not None else 0)

path = sorted((root/"decode").glob("*_rank0.*trace.json.gz"))[0]
events = load_trace(path)
bycorr = collections.defaultdict(list)
for e in events:
    if e.get("cat") in GPU_CATEGORIES and e.get("args",{}).get("correlation") is not None:
        bycorr[e["args"]["correlation"]].append(e)
launches = sorted((e for e in events if e.get("cat") in LAUNCH_CATEGORIES), key=lambda e: e["t"])
S = collections.defaultdict(float)
n = 0
for start, end in step_windows(events):
    corrs = [e["args"]["correlation"] for e in launches
             if start <= e["t"] < end and "correlation" in e.get("args",{})]
    ops = sorted((k for c in corrs for k in bycorr[c]), key=lambda e: e["t"])
    marlin = [e for e in ops if "marlin_moe_wna16" in e["name"]]
    if len(marlin) != 306: continue
    n += 1
    for g in [marlin[i:i+4] for i in range(0, len(marlin), 4)]:
        bys = collections.defaultdict(list)
        for e in g: bys[e["args"].get("stream")].append(e)
        if len(bys) != 2: 
            S["nonfork_cum"] += sum(e["dur"] for e in g)/1000
            S["nonfork_union"] += union(g)
            continue
        a, b = [sum(e["dur"] for e in v)/1000 for v in bys.values()]
        S["fork_cum"] += a + b
        S["fork_floor"] += max(a, b)      # perfect hot/cold overlap
        S["fork_union"] += union(g)
for k in S: S[k] /= n
print(f"decode, rank0, mean over {n} steps (ms/step):")
print(f"  forked groups: cumulative {S['fork_cum']:.3f}  measured union {S['fork_union']:.3f}  "
      f"floor max(hot,cold) {S['fork_floor']:.3f}")
print(f"  realised saving {100*(1-S['fork_union']/S['fork_cum']):.1f}%  of a "
      f"possible {100*(1-S['fork_floor']/S['fork_cum']):.1f}%")
print(f"  remaining headroom = union - floor = {S['fork_union']-S['fork_floor']:.3f} ms/step")
print(f"  non-forked groups: cum {S['nonfork_cum']:.3f} union {S['nonfork_union']:.3f}")
