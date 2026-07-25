"""Host-side call sequence across the MTP draft tail, with CPU op annotations."""
import collections
from tracelib import prep_depth, steps, gpu_ops

ev = prep_depth(0, 3)
st = steps(ev)
a, b = st[1]

rt = sorted([e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")
             and a <= e["t"] < b], key=lambda e: e["t"])
gl = [e for e in rt if "GraphLaunch" in e["name"]]
ops = gpu_ops(ev, a, b)
bycorr = collections.defaultdict(list)
for e in ops:
    bycorr[e["args"].get("correlation")].append(e)
g_end = {}
for g in gl:
    k = bycorr[g["args"]["correlation"]]
    g_end[g["t"]] = (min(e["t"] for e in k), max(e["t"] + e["dur"] / 1000 for e in k))

# cpu_op stack: find the innermost cpu_op covering a timestamp
cpu = sorted([e for e in ev if e.get("cat") == "cpu_op" and a <= e["t"] < b],
             key=lambda e: (e["t"], -e["dur"]))


def owner(t):
    best = None
    for c in cpu:
        if c["t"] > t:
            break
        if c["t"] <= t <= c["t"] + c["dur"] / 1000:
            best = c
    return best["name"] if best else "-"


start = gl[1]["t"]  # after the first draft graph launch
print(f"host calls from first draft-graph launch ({start - a:.3f} ms) onward\n")
print(f"{'t_ms':>9} {'dur_us':>10}  {'api':32s} {'enclosing cpu_op'}")
for e in rt:
    if e["t"] < start - 0.2:
        continue
    if e["dur"] < 20 and "Sync" not in e["name"] and "Graph" not in e["name"]:
        continue
    mark = ""
    if "GraphLaunch" in e["name"]:
        s, en = g_end[e["t"]]
        mark = f"  -> gpu {s - a:.3f}..{en - a:.3f}"
    print(f"{e['t'] - a:9.3f} {e['dur']:10.1f}  {e['name'][:32]:32s} {owner(e['t'])[:46]}{mark}")
