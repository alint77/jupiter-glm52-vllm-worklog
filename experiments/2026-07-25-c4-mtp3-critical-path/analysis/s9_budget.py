"""Full step budget: union span and solo time per kernel category, + all-reduce anatomy."""
import collections, statistics as stat
import tracelib as core
from tracelib import prep_depth, steps, gpu_ops, union_busy

CATS = [
    ("routed W4 Marlin (MoE)", lambda n: n.startswith("void marlin_moe_wna16")),
    ("TP custom all-reduce", lambda n: "cross_device_reduce" in n),
    ("MoE epilogue (act/sum/align)", lambda n: any(t in n for t in
        ("act_and_mul", "moe_sum_vec", "moe_align_block_size", "count_and_sort_expert", "grouped_topk"))),
    ("dense W4 Marlin", lambda n: n.startswith("void marlin::Marlin")),
    ("dense/shared cutlass+nvjet", lambda n: "cutlass::device_kernel" in n or n.startswith("nvjet_")),
    ("sparse MLA (FlashMLA)", lambda n: "sparse_fp8::flash_fwd" in n or "flash_fwd_mla_combine" in n),
    ("DSA indexer (deep_gemm mqa)", lambda n: "deep_gemm::sm90_fp8" in n),
    ("DSA top-k select", lambda n: "cooperative_topk" in n or "StableTopK" in n or "topKPerRow" in n),
    ("DCP NCCL (AG/RS)", lambda n: n.startswith("ncclDevKernel")),
    ("MTP FP8 (deep_gemm gemm)", lambda n: "deep_gemm::fp8_gemm" in n or "fp8_blockscale" in n),
    ("KV cache write", lambda n: "concat_and_cache" in n or "indexer_k_quant" in n),
    ("triton glue", lambda n: n.startswith("triton_")),
    ("elementwise/memset/copy glue", lambda n: any(t in n for t in
        ("elementwise_kernel", "Memset", "Memcpy", "memcpy32_post", "CatArrayBatchedCopy", "reduce_kernel"))),
]

def cat_of(name):
    for label, fn in CATS:
        if fn(name):
            return label
    return "other"

depth = 3
agg_span = collections.Counter(); agg_cum = collections.Counter()
agg_solo = collections.Counter(); agg_n = collections.Counter()
nsteps = 0
ar_durs = []
for rank in range(4):
    ev = prep_depth(rank, depth)
    for a, b in steps(ev):
        ops = gpu_ops(ev, a, b)
        nsteps += 1
        buckets = collections.defaultdict(list)
        for e in ops:
            buckets[cat_of(e["name"])].append(e)
        for c, lst in buckets.items():
            ub, _ = union_busy(lst)
            agg_span[c] += ub
            agg_cum[c] += sum(e["dur"] for e in lst) / 1000
            agg_n[c] += len(lst)
        # solo time: intervals where exactly one category is active
        evs = []
        for c, lst in buckets.items():
            for e in lst:
                evs.append((e["t"], 1, c)); evs.append((e["t"] + e["dur"] / 1000, -1, c))
        evs.sort()
        active = collections.Counter(); prev = None
        for t, d, c in evs:
            if prev is not None and t > prev:
                live = [k for k, v in active.items() if v > 0]
                if len(live) == 1:
                    agg_solo[live[0]] += t - prev
            active[c] += d
            prev = t
        ar_durs += [e["dur"] for e in buckets.get("TP custom all-reduce", [])]

print(f"=== MTP3 c4 step budget, mean over {nsteps} rank-steps (ms/step) ===")
print(f"{'category':32s} {'cum':>8} {'union':>8} {'solo':>8} {'n':>7} {'avg us':>8}")
tot_union = 0
for c, v in agg_span.most_common():
    print(f"{c:32s} {agg_cum[c]/nsteps:8.3f} {v/nsteps:8.3f} {agg_solo[c]/nsteps:8.3f} "
          f"{agg_n[c]/nsteps:7.0f} {1000*agg_cum[c]/agg_n[c]:8.2f}")
    tot_union += v
print(f"{'-'*32} {sum(agg_cum.values())/nsteps:8.3f} {tot_union/nsteps:8.3f} {sum(agg_solo.values())/nsteps:8.3f}")

print("\n=== TP custom all-reduce anatomy ===")
ar_durs.sort()
n = len(ar_durs)
print(f"  calls/step {n/nsteps:.1f}   total {sum(ar_durs)/1000/nsteps:.3f} ms/step")
for p in (5, 25, 50, 75, 90, 99):
    print(f"  p{p:<3d} {ar_durs[int(p/100*(n-1))]:9.2f} us")
print(f"  min {ar_durs[0]:.2f}  max {ar_durs[-1]:.2f}  mean {sum(ar_durs)/n:.2f} us")
floor = ar_durs[int(0.05 * (n - 1))]
print(f"  payload floor (p5) = {floor:.2f} us  ->  wait above floor = "
      f"{(sum(ar_durs) - floor*n)/1000/nsteps:.3f} ms/step ({100*(sum(ar_durs)-floor*n)/sum(ar_durs):.0f}% of all-reduce time)")
