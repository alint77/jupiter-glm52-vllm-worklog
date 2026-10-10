"""Median per-layer kernel chain in the verify graph, anchored at each
grouped_topk launch, over layers with the most common kernel sequence: start /
end offsets (us) from the anchor, the gap since all earlier kernels ended
(negative = overlap) and the duration.

    layer_chain.py <trace.json.gz>
"""
import collections
import gzip
import json
import statistics as st
import sys

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
ks = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
anchors = [i for i, k in enumerate(ks) if "grouped_topk" in k["name"]]
segs = collections.defaultdict(list)
for a, b in zip(anchors, anchors[1:]):
    if ks[b]["ts"] - ks[a]["ts"] > 1000:  # crosses a step boundary
        continue
    segs[tuple(k["name"] for k in ks[a:b])].append(ks[a:b + 1])
seq, layers = max(segs.items(), key=lambda x: len(x[1]))
print(f"{len(layers)} of {sum(map(len, segs.values()))} layers share the most common sequence ({len(seq)} kernels)")
print(f"{'start':>7} {'end':>7} {'gap':>6} {'dur':>6}  stream kernel")
cols = []
for seg in layers:
    t0, prev, row = seg[0]["ts"], seg[0]["ts"], []
    for k in seg:
        row.append((k["ts"] - t0, k["ts"] + k["dur"] - t0, k["ts"] - prev, k["dur"]))
        prev = max(prev, k["ts"] + k["dur"])
    cols.append(row)
for j, k in enumerate(layers[0]):
    med = [st.median(c[j][i] for c in cols) for i in range(4)]
    name = "(next layer's grouped_topk)" if j == len(seq) else k["name"][:80]
    print(f"{med[0]:7.1f} {med[1]:7.1f} {med[2]:6.1f} {med[3]:6.1f}  {k['args'].get('stream'):>4} {name}")
