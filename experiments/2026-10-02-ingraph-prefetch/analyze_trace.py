"""Per layer: cold copy (L+1) vs MoE(L) span, and compute idle at the join.

Graph replays report every node on the launch stream, so copies are told apart
from compute by event type: the cold copies are the >=5 MB HtoD at C2C speed.
Usage: analyze_trace.py <trace.json.gz>
"""
import gzip, json, sys, statistics as st

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
kern = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
copies = sorted((e for e in ev if e.get("cat") == "gpu_memcpy"
                 and "HtoD" in e["name"] and e["args"].get("bytes", 0) >= 5 << 20
                 and e["args"].get("memory bandwidth (GB/s)", 0) < 600),
                key=lambda e: e["ts"])
end = lambda e: e["ts"] + e["dur"]

# Bursts: one layer's components go back to back on the copy stream.
bursts = []
for c in copies:
    if bursts and c["ts"] - bursts[-1][1] < 20:
        bursts[-1][1] = end(c); bursts[-1][2] += c["args"]["bytes"]
    else:
        bursts.append([c["ts"], end(c), c["args"]["bytes"]])

# MoE spans: route_kernel .. combine_kernel (wgmma prefill MoE), else
# moe_align .. moe_sum, two of each per layer (hot, cold Marlin tiers).
moes, cur = [], None
for e in kern:
    if "route_kernel" in e["name"]:
        cur = [e["ts"], None]
    elif "combine_kernel" in e["name"] and cur is not None:
        cur[1] = end(e); moes.append(tuple(cur)); cur = None
if not moes:
    marks = [e for e in kern if "moe_align_block_size" in e["name"] or "moe_sum" in e["name"]]
    for e in marks:
        if "moe_align" in e["name"]:
            if cur is None or cur[2] == 2:
                cur = [e["ts"], None, 0]; moes.append(cur)
        else:
            cur[1] = end(e); cur[2] += 1
    moes = [(s, t) for s, t, n in moes if t is not None]

# Prefills: split MoE spans on >5 ms gaps.
groups, g = [], []
for m in moes:
    if g and m[0] - g[-1][1] > 5000:
        groups.append(g); g = []
    g.append(m)
groups.append(g)

def idle_before(t):
    """Compute idle gap ending at the first kernel that starts after time t."""
    nxt = next((k for k in kern if k["ts"] >= t), None)
    if nxt is None: return 0.0
    prev_end = max((end(k) for k in kern if k["ts"] < nxt["ts"]), default=nxt["ts"])
    return max(0.0, nxt["ts"] - prev_end)

for gi, g in enumerate(groups):
    t0, t1 = g[0][0], g[-1][1]
    bs = [b for b in bursts if t0 - 3000 <= b[0] <= t1]
    rows = []
    for b in bs:
        # MoE(L): the first MoE that ends after the copy starts; skip the
        # dense-layer copy, whose next MoE is the layer being copied.
        nxt = next((m for m in g if m[1] > b[0]), None)
        if nxt is None: continue
        dense = nxt[0] > b[0] and nxt is g[0] and b[0] < g[0][0]
        rows.append(dict(copy=b[1] - b[0], mb=b[2] / 2**20, dense=dense,
                         moe=nxt[1] - nxt[0], slack=nxt[1] - b[1],
                         idle=idle_before(b[1])))
    body = [r for r in rows if not r["dense"]]
    late = [r for r in body if r["slack"] < 0]
    print(f"prefill {gi}: {len(g)} MoE layers, {len(rows)} copies, wall {(t1-t0)/1e3:.1f} ms (first..last MoE)")
    print(f"  copy us  median {st.median(r['copy'] for r in body):.0f}  max {max(r['copy'] for r in body):.0f}"
          f"   MB median {st.median(r['mb'] for r in body):.0f}")
    print(f"  MoE(L) us median {st.median(r['moe'] for r in body):.0f}  min {min(r['moe'] for r in body):.0f}")
    print(f"  copy ends after MoE(L): {len(late)}/{len(body)} layers, by median "
          f"{st.median([-r['slack'] for r in late]) if late else 0:.0f} us, max {max([-r['slack'] for r in late], default=0):.0f} us")
    print(f"  copy longer than MoE(L) (MoE-only window too short): "
          f"{sum(r['copy'] > r['moe'] for r in body)}/{len(body)}")
    print(f"  compute idle right at copy end (join stall) >5us: {sum(r['idle']>5 for r in body)} layers, "
          f"total {sum(r['idle'] for r in body if r['idle']>5):.0f} us")
    d = [r for r in rows if r["dense"]]
    if d: print(f"  dense-layer copy: {d[0]['copy']:.0f} us, idle at its end {d[0]['idle']:.0f} us")
