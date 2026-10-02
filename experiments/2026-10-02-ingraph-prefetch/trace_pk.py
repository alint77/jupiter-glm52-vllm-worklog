"""Per-layer MoE span with the wgmma prefill MoE (route_kernel .. combine_kernel)
vs Marlin (moe_align .. second moe_sum), the cold copy of L+1 against it, and
the forward span, per traced prefill (512 / 768 / 1024 new tokens).
Usage: trace_pk.py <trace.json.gz> [marlin trace.json.gz]
"""
import gzip, json, statistics as st, sys


def load(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    kern = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    copies = sorted((e for e in ev if e.get("cat") == "gpu_memcpy"
                     and "HtoD" in e["name"] and e["args"].get("bytes", 0) >= 5 << 20
                     and e["args"].get("memory bandwidth (GB/s)", 0) < 600),
                    key=lambda e: e["ts"])
    return kern, copies


end = lambda e: e["ts"] + e["dur"]


def moe_spans(kern):
    out, cur = [], None
    for e in kern:
        n = e["name"]
        if "route_kernel" in n:
            cur = [e["ts"], None, "pk"]
        elif "combine_kernel" in n and cur is not None:
            cur[1] = end(e); out.append(cur); cur = None
    if out:
        return out
    marks = [e for e in kern if "moe_align_block_size" in e["name"] or "moe_sum" in e["name"]]
    cur = None
    for e in marks:
        if "moe_align" in e["name"]:
            if cur is None or cur[2] == 2:
                cur = [e["ts"], None, 0]; out.append(cur)
        else:
            cur[1] = end(e); cur[2] += 1
    return [m for m in out if m[1] is not None]


def bursts(copies):
    b = []
    for c in copies:
        if b and c["ts"] - b[-1][1] < 20:
            b[-1][1] = end(c)
        else:
            b.append([c["ts"], end(c)])
    return b


for path in sys.argv[1:]:
    kern, copies = load(path)
    moes = moe_spans(kern)
    groups, g = [], []
    for m in moes:
        if g and m[0] - g[-1][1] > 5000:
            groups.append(g); g = []
        g.append(m)
    groups.append(g)
    cb = bursts(copies)
    print(path.split("/")[-2], "kind", moes[0][2] if moes else None)
    for g in groups[-3:]:
        d = [(m[1] - m[0]) for m in g]
        # copy of the layer after each MoE: the burst that ends latest before
        # the next MoE starts, and whether it ends after that MoE does
        late = 0
        for a, b in zip(g, g[1:]):
            inside = [c for c in cb if a[0] - 3000 < c[0] < b[0]]
            if inside and inside[-1][1] > a[1]:
                late += 1
        print(f"  {len(g)} MoE layers, MoE median {st.median(d):7.0f} us "
              f"(p10 {sorted(d)[len(d)//10]:.0f}, p90 {sorted(d)[9*len(d)//10]:.0f}), "
              f"sum {sum(d)/1000:6.1f} ms, first->last MoE {(g[-1][1]-g[0][0])/1000:6.1f} ms, "
              f"layers whose next copy outlasts MoE {late}")
