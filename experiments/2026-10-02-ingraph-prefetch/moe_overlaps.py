"""Which kernels run concurrently with the MoE expert GEMMs (Marlin)?
moe_overlaps.py <trace> <tokens>"""
import gzip, json, sys, collections
end = lambda e: e["ts"] + e["dur"]
ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
ann = [e for e in ev if e.get("cat") == "gpu_user_annotation"
       and e["name"].startswith(f"execute_context_1({sys.argv[2]})")]
W0 = min(e["ts"] for e in ann); W1 = max(end(e) for e in ann)
k = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")
            and W0 <= e["ts"] < W1), key=lambda e: e["ts"])
mar = [e for e in k if "marlin" in e["name"]]
# MoE window per layer = first..last Marlin of the layer (4 per layer).
wins = [(mar[i]["ts"], end(mar[i + 3])) for i in range(0, len(mar) - 3, 4)]
over = collections.defaultdict(lambda: [0, 0.0])
for a, b in wins:
    for e in k:
        if "marlin" in e["name"] or end(e) <= a or e["ts"] >= b:
            continue
        o = min(end(e), b) - max(e["ts"], a)
        kind = e["cat"]
        name = e["name"][:70]
        if kind == "gpu_memcpy":
            name = f"{e['name']} ({'>=5MB' if e['args'].get('bytes',0) >= 5<<20 else 'small'})"
        over[name][0] += 1; over[name][1] += o
tot = sum(b - a for a, b in wins)
print(f"{sys.argv[2]} tokens: {len(wins)} MoE windows, {tot/1e3:.1f} ms total")
for n, (c, t) in sorted(over.items(), key=lambda kv: -kv[1][1]):
    print(f"  {c:4d}x  {t/1e3:7.2f} ms overlap  {n}")
