"""Break one traced prefill down: GPU idle, where it sits, kernel categories,
one layer's kernel sequence with overlaps.  breakdown.py <trace> <tokens> [layer]"""
import gzip, json, sys, collections, bisect

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
tokens, show = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40
ann = [e for e in ev if e.get("cat") == "gpu_user_annotation"
       and e["name"].startswith(f"execute_context_1({tokens})")]
W0 = min(e["ts"] for e in ann); W1 = max(e["ts"] + e["dur"] for e in ann)
end = lambda e: e["ts"] + e["dur"]
inw = lambda e: W0 <= e["ts"] < W1
launch = {e["args"]["correlation"]: e for e in ev
          if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}

def is_cold(e):
    return (e["cat"] == "gpu_memcpy" and "HtoD" in e["name"]
            and e["args"].get("bytes", 0) >= 5 << 20)

work = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
               and inw(e) and not is_cold(e)), key=lambda e: e["ts"])
cold = sorted((e for e in ev if e.get("cat") == "gpu_memcpy" and inw(e) and is_cold(e)),
              key=lambda e: e["ts"])

def cat(n):
    n = n.lower()
    for key, c in [("marlin", "MoE marlin"), ("gemm_kernel", "MoE wgmma"),
                   ("route_kernel", "MoE routing"), ("gather_kernel", "MoE routing"),
                   ("act_kernel", "MoE act"), ("combine_kernel", "MoE routing"), ("moe_align", "MoE routing"), ("count_and_sort", "MoE routing"),
                   ("grouped_topk", "MoE routing"), ("moe_sum", "MoE routing"), ("act_and_mul", "MoE act"),
                   ("nccl", "collective"), ("cross_device_reduce", "collective"), ("allreduce", "collective"),
                   ("flash", "attention"), ("flashmla", "attention"), ("mla", "attention"), ("indexer", "indexer"),
                   ("fp8_mqa", "indexer"), ("top_k", "indexer"), ("topk", "indexer"),
                   ("nvjet", "GEMM"), ("gemm", "GEMM"), ("cutlass", "GEMM"), ("splitk", "GEMM"),
                   ("memcpy", "copy"), ("triton", "triton fused"), ("elementwise", "torch elementwise"),
                   ("fill", "torch elementwise")]:
        if key in n: return c
    return "other"

# Busy union and gaps.
busy, gaps, last_end, last = 0.0, [], W0, None
for e in work:
    s, t = e["ts"], end(e)
    if s > last_end:
        gaps.append((s - last_end, last, e))
    busy += max(0.0, t - max(s, last_end))
    if t > last_end: last_end, last = t, e
wall = W1 - W0
idle = wall - busy
print(f"== {tokens}-token prefill, rank file {sys.argv[1].split('/')[-1][:30]}")
print(f"window {wall/1e3:.2f} ms, compute busy {busy/1e3:.2f} ms, idle {idle/1e3:.2f} ms ({100*idle/wall:.0f}%), "
      f"{len(work)} kernels/ops, {len(cold)} cold copies")

def host_bound(prev, nxt):
    api = launch.get(nxt["args"].get("correlation"))
    return api is not None and prev is not None and end(api) > end(prev)

hist = collections.Counter()
by_pair = collections.defaultdict(lambda: [0, 0.0, 0])
for g, prev, nxt in gaps:
    b = "<5us" if g < 5 else "5-20us" if g < 20 else "20-100us" if g < 100 else ">100us"
    hist[b] += g
    k = (cat(prev["name"]) if prev else "start", cat(nxt["name"]))
    by_pair[k][0] += 1; by_pair[k][1] += g; by_pair[k][2] += host_bound(prev, nxt)
print("idle by gap size (ms):", {k: round(v / 1e3, 2) for k, v in sorted(hist.items())})
print("idle by transition (prev -> next): count, total ms, host-bound count")
for k, (n, g, h) in sorted(by_pair.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f"  {k[0]:>18} -> {k[1]:<18} {n:5d} {g/1e3:7.2f} ms  host-bound {h}")
hb = sum(g for g, p, n in gaps if host_bound(p, n))
print(f"idle where the next launch came after the GPU went idle (host-bound): {hb/1e3:.2f} ms")

cats = collections.defaultdict(float)
for e in work: cats[cat(e["name"])] += e["dur"]
print("kernel time by category (ms, sum of durations):")
for c, v in sorted(cats.items(), key=lambda kv: -kv[1]): print(f"  {c:<18} {v/1e3:7.2f}")
cb = sum(e["dur"] for e in cold)
print(f"cold copies: {cb/1e3:.2f} ms of copy-engine time, {sum(e['args']['bytes'] for e in cold)/2**30:.2f} GiB")

# One layer: from the end of MoE(show-1) to the end of MoE(show).
comb = [e for e in work if "combine_kernel" in e["name"]]
if comb:
    ends = [end(e) for e in comb]
else:
    sums = [e for e in work if "moe_sum" in e["name"]]
    ends = [end(sums[2 * i + 1]) for i in range(len(sums) // 2)]
a, b = ends[show - 1], ends[show]
print(f"\n== layer sequence: after MoE #{show-1} to end of MoE #{show} ({(b-a)/1e3:.2f} ms)")
print(f"{'t(us)':>7} {'dur':>6} {'gap':>5} {'cat':<16} name")
prev_end = a
for e in work:
    if not (a <= e["ts"] < b): continue
    g = e["ts"] - prev_end
    flag = " H" if g > 5 and host_bound(None if False else {"ts":0,"dur":prev_end}, e) else ""
    print(f"{e['ts']-a:7.0f} {e['dur']:6.0f} {g:5.0f} {cat(e['name']):<16} {e['name'][:70]}{flag}")
    prev_end = max(prev_end, end(e))
for c in cold:
    if end(c) > a and c["ts"] < b:
        print(f"   copy {c['ts']-a:7.0f}..{end(c)-a:7.0f}  {c['args']['bytes']/2**20:5.0f} MB")
