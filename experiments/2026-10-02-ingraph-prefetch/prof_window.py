"""Per-step GPU breakdown over the step windows (gpu_user_annotation
execute_*) whose name contains PATTERN: wall, busy, idle (and how much of it
waited on a host launch), kernel time by category, top kernels; per step
(averaged when several steps match). Cold-tier copies (>= 5 MB HtoD) are
reported apart from compute.
Usage: prof_window.py <trace.json.gz> <pattern> [top_n] [--list]
"""
import collections, gzip, json, sys

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
pat = sys.argv[2]
top_n = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 20
ann = [e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute")]
if "--list" in sys.argv:
    for n, c in collections.Counter(e["name"] for e in ann).most_common():
        print(f"{c:5d}  {n}")
    sys.exit()
wins = sorted(((e["ts"], e["ts"] + e["dur"]) for e in ann if pat in e["name"]))
# a step can be annotated on several streams: merge overlapping windows
merged = []
for w in wins:
    if merged and w[0] < merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], w[1]))
    else:
        merged.append(w)
wins = merged
end = lambda e: e["ts"] + e["dur"]
launch = {e["args"]["correlation"]: e for e in ev
          if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
             key=lambda e: e["ts"])
cold = lambda e: (e["cat"] == "gpu_memcpy" and "HtoD" in e["name"]
                  and e["args"].get("bytes", 0) >= 5 << 20)

CATS = [("tiered_decode", "MoE decode"), ("gemm_kernel<", "MoE prefill (wgmma)"),
        ("tiered_prefill::", "MoE prefill glue"), ("marlin", "MoE marlin"),
        ("moe_align", "MoE glue"), ("moe_sum", "MoE glue"), ("count_and_sort", "MoE glue"),
        ("grouped_topk", "router"), ("topk_softmax", "router"),
        ("sparse_fp8", "sparse MLA"), ("sparse_attn_fwd", "sparse MLA"),
        ("upconvert", "copies/cat"), ("flash_fwd_splitkv_mla", "MLA decode"),
        ("flashmla", "MLA"), ("FlashAttnFwd", "dense attention"),
        ("indexer", "indexer"), ("fp8_mqa", "indexer"), ("mqa_logits", "indexer"),
        ("TopK", "indexer topk"), ("topk", "indexer topk"),
        ("_correct_attn_cp_out", "DCP combine"), ("dcp", "DCP"),
        ("nccl", "collective"), ("cross_device_reduce", "collective"),
        ("allreduce", "collective"), ("one_shot", "collective"), ("lamport", "collective"),
        ("nvjet", "dense GEMM"), ("cutlass", "dense GEMM"), ("gemm", "dense GEMM"),
        ("skinny", "dense GEMM"),
        ("rms_norm", "norm"), ("rotary", "rope"), ("rope", "rope"),
        ("triton", "triton fused"), ("CatArray", "copies/cat"), ("direct_copy", "copies/cat"),
        ("Memcpy", "copies/cat"), ("Memset", "copies/cat"), ("fill", "elementwise"),
        ("elementwise", "elementwise"), ("reduce_kernel", "elementwise")]

def cat(n):
    for k, c in CATS:
        if k in n:
            return c
    return "other"

tot = collections.defaultdict(float); spans = collections.defaultdict(list); top = collections.defaultdict(lambda: [0, 0.0])
wall = busy = hostb = coldt = 0.0
for w0, w1 in wins:
    work = [e for e in gpu if w0 <= e["ts"] < w1]
    last_end, last = w0, None
    for e in work:
        if cold(e):
            coldt += e["dur"]; continue
        s, t = e["ts"], end(e)
        if s > last_end:
            api = launch.get(e["args"].get("correlation"))
            if api is not None and end(api) > last_end:
                hostb += s - max(last_end, api["ts"] if api["ts"] > last_end else last_end)
        busy += max(0.0, t - max(s, last_end))
        if t > last_end:
            last_end, last = t, e
        tot[cat(e["name"])] += e["dur"]
        spans[cat(e["name"])].append((s, t))
        a = top[e["name"][:100]]; a[0] += 1; a[1] += e["dur"]
    wall += w1 - w0
k = len(wins)
print(f"{k} window(s) matching {pat!r}; per window: wall {wall / k / 1e3:.2f} ms, "
      f"busy {busy / k / 1e3:.2f}, idle {(wall - busy) / k / 1e3:.2f} "
      f"({100 * (wall - busy) / wall:.0f}%), of which waiting on a host launch "
      f"{hostb / k / 1e3:.2f}; cold copies {coldt / k / 1e3:.2f} ms")
def union(iv):
    u, cur = 0.0, None
    for a, b in sorted(iv):
        if cur is None or a > cur[1]:
            if cur: u += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return u + (cur[1] - cur[0] if cur else 0.0)
print("by category, ms per window: union (time any kernel of it runs) | sum of durations")
for c, v in sorted(tot.items(), key=lambda kv: -union(spans[kv[0]])):
    print(f"  {c:<22} {union(spans[c]) / k / 1e3:8.3f} | {v / k / 1e3:8.3f}")
print(f"top {top_n} kernels, per window:")
for n, (c, d) in sorted(top.items(), key=lambda kv: -kv[1][1])[:top_n]:
    print(f"  {d / k / 1e3:8.3f} ms {c / k:7.1f} x {d / c:8.1f} us  {n}")
