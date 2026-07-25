"""Anatomy of the eager prologue and the MTP draft tail (host-latency regions)."""
import collections
from tracelib import prep_depth, steps, gpu_ops, union_busy, gaps

depth = 3
ev = prep_depth(0, depth)
st = steps(ev)

agg = collections.Counter(); aggn = collections.Counter()
pro_wall = pro_idle = tail_wall = tail_idle = 0.0
nst = 0
for a, b in st:
    rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and a <= e["t"] < b]
    gl = sorted([e for e in rt if "GraphLaunch" in e["name"]], key=lambda e: e["t"])
    ops = gpu_ops(ev, a, b)
    corrs = [g["args"]["correlation"] for g in gl]
    tgt = [e for e in ops if e["args"].get("correlation") == corrs[0]]
    ts = min(e["t"] for e in tgt); te = max(e["t"] + e["dur"] / 1000 for e in tgt)
    nst += 1

    pro = [e for e in ops if e["t"] < ts]
    ub, mg = union_busy(pro); pro_wall += ts - a; pro_idle += (ts - a) - ub
    for e in pro:
        agg[e["name"][:52]] += e["dur"]; aggn[e["name"][:52]] += 1
    tail = [e for e in ops if e["t"] >= te]
    ub2, mg2 = union_busy(tail); tail_wall += b - te; tail_idle += (b - te) - ub2

print(f"=== eager PROLOGUE (before target graph), mean of {nst} steps ===")
print(f"  wall {pro_wall/nst:.3f} ms   GPU idle {pro_idle/nst:.3f} ms  ({100*pro_idle/pro_wall:.0f}% idle)")
print(f"  {'kernel':54s} {'n/step':>7} {'ms/step':>9}")
for n, v in agg.most_common(18):
    print(f"  {n:54s} {aggn[n]/nst:7.1f} {v/1000/nst:9.4f}")

print(f"\n=== MTP DRAFT TAIL (after target graph), mean of {nst} steps ===")
print(f"  wall {tail_wall/nst:.3f} ms   GPU idle {tail_idle/nst:.3f} ms  ({100*tail_idle/tail_wall:.0f}% idle)")

# draft graph spans + gaps for one step
a, b = st[1]
rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and a <= e["t"] < b]
gl = sorted([e for e in rt if "GraphLaunch" in e["name"]], key=lambda e: e["t"])
ops = gpu_ops(ev, a, b)
bycorr = collections.defaultdict(list)
for e in ops:
    bycorr[e["args"].get("correlation")].append(e)
print(f"\n  step1 detail:")
prev_end = None
for gi, g in enumerate(gl):
    k = bycorr[g["args"]["correlation"]]
    s = min(e["t"] for e in k); en = max(e["t"] + e["dur"] / 1000 for e in k)
    if prev_end is not None:
        eager = [e for e in ops if prev_end <= e["t"] < s]
        ub, _ = union_busy(eager) if eager else (0.0, None)
        hostcalls = [e for e in rt if prev_end <= e["t"] < s]
        print(f"    GAP {prev_end-a:7.3f}->{s-a:7.3f} = {(s-prev_end)*1000:7.1f} us | "
              f"{len(eager):3d} eager gpu ops ({ub*1000:6.1f} us busy) | {len(hostcalls):3d} host API calls")
    print(f"    {'TARGET' if gi==0 else f'DRAFT{gi}':7s} gpu {s-a:7.3f}->{en-a:7.3f} span={(en-s):7.3f} ms nodes={len(k):5d}")
    prev_end = en
print(f"    GAP {prev_end-a:7.3f}->{b-a:7.3f} = {(b-prev_end)*1000:7.1f} us  (into next step's prologue)")

# host blocking calls
sync = [e for e in rt if "Synchronize" in e["name"] or "EventQuery" in e["name"]]
print(f"\n  host blocking/poll calls in step: {collections.Counter(e['name'] for e in sync)}")
for e in sorted(sync, key=lambda x: x["t"]):
    if e["dur"] > 20:
        print(f"    {e['t']-a:8.3f} +{e['dur']:9.1f}us {e['name']}")
