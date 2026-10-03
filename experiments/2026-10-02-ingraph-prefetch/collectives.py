"""Collectives and copies in one traced step, by the op that launched them:
count, mean duration, operand size (from record_shapes) and bandwidth.
Usage: collectives.py <trace.json.gz> <annotation prefix>"""
import collections, gzip, json, math, sys

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
ann = [e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith(sys.argv[2])]
w0 = min(e["ts"] for e in ann); w1 = max(e["ts"] + e["dur"] for e in ann)
rt = {e["args"]["correlation"]: e for e in ev
      if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
ops = collections.defaultdict(list)
for e in ev:
    if e.get("cat") == "cpu_op":
        ops[e["tid"]].append(e)
for t in ops:
    ops[t].sort(key=lambda e: e["ts"])

def chain(api):
    return [e for e in ops.get(api["tid"], []) if e["ts"] <= api["ts"] <= e["ts"] + e["dur"]]

def dims(op):
    d = op.get("args", {}).get("Input Dims") or []
    t = op.get("args", {}).get("Input type") or []
    return [(tuple(x), y) for x, y in zip(d, t)
            if isinstance(x, list) and x and all(isinstance(v, int) for v in x)]

SIZE = {"c10::BFloat16": 2, "float": 4, "c10::Half": 2, "int": 4, "long int": 8,
        "unsigned char": 1, "c10::Float8_e4m3fn": 1}
agg = collections.defaultdict(lambda: [0, 0.0, None])
for k in ev:
    if k.get("cat") != "kernel" or not (w0 <= k["ts"] < w1):
        continue
    n = k["name"]
    kind = next((s for s in ("AllGather", "ReduceScatter", "AllReduce", "one_shot", "direct_copy",
                             "CatArray", "lamport", "cross_device") if s in n), None)
    if kind is None:
        continue
    api = rt.get(k["args"].get("correlation"))
    ch = chain(api) if api else []
    names = [c["name"] for c in ch]
    ctx = next((x for x in names if x in ("vllm::unified_mla_attention_with_output",
                                          "vllm::sparse_attn_indexer", "vllm::moe_forward",
                                          "vllm::moe_forward_shared")), "")
    op = next((c for c in reversed(ch) if c["name"].startswith(("vllm::all", "vllm::reduce",
               "vllm::dcp", "aten::cat", "aten::copy_", "aten::clone", "aten::contiguous"))), None)
    opn = op["name"] if op else (names[-1] if names else "?")
    d = dims(op) if op else []
    nbytes = SIZE.get(d[0][1], 2) * math.prod(d[0][0]) if d else 0
    key = (kind, ctx.replace("vllm::", ""), opn, d[0][0] if d else ())
    a = agg[key]; a[0] += 1; a[1] += k["dur"]; a[2] = nbytes
print(f"{'kind':<13} {'context':<34} {'op':<16} {'input shape':<22} {'n':>4} {'ms':>7} {'us/call':>8} {'MB':>7} {'GB/s':>6}")
for (kind, ctx, opn, shp), (c, d, b) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    us = d / c
    print(f"{kind:<13} {ctx[:34]:<34} {opn[:16]:<16} {str(shp)[:22]:<22} {c:4d} {d/1e3:7.2f} {us:8.1f} "
          f"{b/1e6:7.1f} {b/us/1e3 if b else 0:6.0f}")
