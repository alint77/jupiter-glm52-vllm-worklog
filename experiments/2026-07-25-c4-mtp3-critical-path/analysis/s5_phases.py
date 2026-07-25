import collections
from tracelib import prep, steps, gpu_ops, union_busy, gaps

for rank in [0]:
    ev = prep(rank)
    st = steps(ev)
    for si, (a, b) in enumerate(st):
        rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and a <= e["t"] < b]
        gl = sorted([e for e in rt if "GraphLaunch" in e["name"]], key=lambda e: e["t"])
        ops = gpu_ops(ev, a, b)
        bycorr = collections.defaultdict(list)
        for e in ops:
            bycorr[e["args"].get("correlation")].append(e)
        print(f"\n=== rank{rank} step{si}  wall={b-a:.3f} ms, {len(gl)} graph launches, {len(ops)} gpu ops")
        covered = 0
        for g in gl:
            k = bycorr.get(g["args"]["correlation"], [])
            if not k:
                continue
            s = min(e["t"] for e in k)
            en = max(e["t"] + e["dur"] / 1000 for e in k)
            ub, merged = union_busy(k)
            covered += len(k)
            print(f"  GRAPH corr={g['args']['correlation']} host_t={g['t']-a:7.3f} host_dur={g['dur']:8.1f}us "
                  f"| gpu {s-a:7.3f}->{en-a:7.3f} span={en-s:7.3f} nodes={len(k):5d} busy={ub:7.3f} idle={en-s-ub:6.3f}")
        eager = [e for e in ops if e["args"].get("correlation") not in {g["args"]["correlation"] for g in gl}]
        eub, _ = union_busy(eager)
        print(f"  EAGER (non-graph) gpu ops: {len(eager)}  cum={sum(e['dur'] for e in eager)/1000:.3f} ms union={eub:.3f} ms")
        # gaps
        ub, merged = union_busy(ops)
        gg = gaps(merged, a, b)
        big = sorted(gg, key=lambda x: -(x[1] - x[0]))[:12]
        tot_idle = sum(y - x for x, y in gg)
        print(f"  union_busy={ub:.3f} idle={tot_idle:.3f} over {len(gg)} gaps; largest:")
        for x, y in big:
            print(f"     {x-a:8.3f} -> {y-a:8.3f}  ({(y-x)*1000:8.1f} us)")
        if si >= 1:
            break
