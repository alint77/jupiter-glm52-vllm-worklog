"""Per-layer MoE tier decomposition.

Tier id: cold runs on aux_stream (forked first), hot on the main stream.
`cold_output.add_(hot_output)` runs on the MAIN stream after the join, so the
stream carrying the CUDAFunctor_add right after the two moe_sum_vec is HOT.
"""
import collections, statistics as stat, json, sys
from tracelib import prep, steps, gpu_ops, union_busy

def layer_table(ev, a, b):
    ops = gpu_ops(ev, a, b)
    anchors = sorted([e for e in ops if "grouped_topk" in e["name"]], key=lambda e: e["t"])
    rows = []
    for i, an in enumerate(anchors):
        hi = anchors[i + 1]["t"] if i + 1 < len(anchors) else b
        seg = [e for e in ops if an["t"] <= e["t"] < hi]
        mar = [e for e in seg if e["name"].startswith("void marlin_moe_wna16")]
        if len(mar) != 4:
            continue  # MTP draft layers use deep_gemm, not marlin
        adds = [e for e in seg if "CUDAFunctor_add<c10::BFloat16>" in e["name"]]
        sums = [e for e in seg if "moe_sum_vec" in e["name"]]
        if not adds or len(sums) != 2:
            continue
        add = min(adds, key=lambda e: e["t"])
        hot_str = add["args"]["stream"]
        strs = {e["args"]["stream"] for e in mar}
        if hot_str not in strs or len(strs) != 2:
            continue
        cold_str = (strs - {hot_str}).pop()
        hot = sorted([e for e in mar if e["args"]["stream"] == hot_str], key=lambda e: e["t"])
        cold = sorted([e for e in mar if e["args"]["stream"] == cold_str], key=lambda e: e["t"])
        if len(hot) != 2 or len(cold) != 2:
            continue
        acts = [e for e in seg if "act_and_mul" in e["name"]]
        hot_act = [e for e in acts if e["args"]["stream"] == hot_str]
        cold_act = [e for e in acts if e["args"]["stream"] == cold_str]
        chain = [e for e in seg if e["args"]["stream"] in (hot_str, cold_str)]
        span_lo = min(e["t"] for e in mar)
        span_hi = max(e["t"] + e["dur"] / 1000 for e in mar + sums)
        rows.append(dict(
            layer=len(rows),
            hot_w13=hot[0]["dur"], hot_w2=hot[1]["dur"],
            cold_w13=cold[0]["dur"], cold_w2=cold[1]["dur"],
            hot_act=hot_act[0]["dur"] if hot_act else 0.0,
            cold_act=cold_act[0]["dur"] if cold_act else 0.0,
            hot_start=hot[0]["t"], cold_start=cold[0]["t"],
            hot_end=hot[1]["t"] + hot[1]["dur"] / 1000,
            cold_end=cold[1]["t"] + cold[1]["dur"] / 1000,
            span=(span_hi - span_lo) * 1000,
            marlin_sum=sum(e["dur"] for e in mar),
        ))
    return rows

if __name__ == "__main__":
    ev = prep(0)
    st = steps(ev)
    allrows = []
    for a, b in st:
        allrows.append(layer_table(ev, a, b))
    print("layers found per step:", [len(r) for r in allrows])
    rows = allrows[1]
    print(f"\n{'ly':>3} {'hotW13':>8} {'hotAct':>7} {'hotW2':>8} {'HOTtot':>8} | "
          f"{'cldW13':>8} {'cldAct':>7} {'cldW2':>8} {'CLDtot':>8} | {'span':>8} {'sum':>8} {'save%':>6} {'coldFinFirst':>6}")
    tot = collections.Counter()
    for r in rows:
        h = r["hot_w13"] + r["hot_act"] + r["hot_w2"]
        c = r["cold_w13"] + r["cold_act"] + r["cold_w2"]
        save = 100 * (1 - r["span"] / r["marlin_sum"])
        cf = "cold" if r["cold_end"] < r["hot_end"] else "HOT"
        if r["layer"] < 20 or r["layer"] > 70:
            print(f"{r['layer']:3d} {r['hot_w13']:8.1f} {r['hot_act']:7.1f} {r['hot_w2']:8.1f} {h:8.1f} | "
                  f"{r['cold_w13']:8.1f} {r['cold_act']:7.1f} {r['cold_w2']:8.1f} {c:8.1f} | "
                  f"{r['span']:8.1f} {r['marlin_sum']:8.1f} {save:6.1f} {cf:>6}")
        tot["hot"] += h; tot["cold"] += c; tot["span"] += r["span"]; tot["sum"] += r["marlin_sum"]
        tot["hw13"] += r["hot_w13"]; tot["hw2"] += r["hot_w2"]
        tot["cw13"] += r["cold_w13"]; tot["cw2"] += r["cold_w2"]
        tot["hact"] += r["hot_act"]; tot["cact"] += r["cold_act"]
        tot["coldfirst"] += 1 if r["cold_end"] < r["hot_end"] else 0
        tot["n"] += 1
    n = tot["n"]
    print(f"\n=== step totals over {n} routed layers (ms) ===")
    print(f"  hot chain  : {tot['hot']/1000:8.3f}   (w13 {tot['hw13']/1000:6.3f} act {tot['hact']/1000:6.3f} w2 {tot['hw2']/1000:6.3f})")
    print(f"  cold chain : {tot['cold']/1000:8.3f}   (w13 {tot['cw13']/1000:6.3f} act {tot['cact']/1000:6.3f} w2 {tot['cw2']/1000:6.3f})")
    print(f"  marlin sum : {tot['sum']/1000:8.3f}")
    print(f"  layer spans: {tot['span']/1000:8.3f}  -> overlap saving {100*(1-tot['span']/tot['sum']):.1f}%")
    print(f"  cold finishes first in {tot['coldfirst']}/{n} layers ({100*tot['coldfirst']/n:.0f}%)")
