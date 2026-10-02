"""Top kernels by summed duration inside one traced prefill's window, with
counts and per-call size.  top_kernels.py <trace> <tokens> [n]"""
import collections, gzip, json, sys

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
tok, n = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 25
ann = [e for e in ev if e.get("cat") == "gpu_user_annotation"
       and e["name"].startswith(f"execute_context_1({tok})")]
w0 = min(e["ts"] for e in ann); w1 = max(e["ts"] + e["dur"] for e in ann)
agg = collections.defaultdict(lambda: [0, 0.0])
for e in ev:
    if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and w0 <= e["ts"] < w1:
        a = agg[e["name"][:110]]; a[0] += 1; a[1] += e["dur"]
print(f"== {tok} tokens, window {(w1 - w0) / 1e3:.1f} ms")
for k, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:n]:
    print(f"{d / 1e3:8.2f} ms {c:6d} x {d / c:8.1f} us  {k}")
