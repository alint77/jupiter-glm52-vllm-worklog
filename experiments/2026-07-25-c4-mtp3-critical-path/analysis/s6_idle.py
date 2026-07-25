import collections
from tracelib import prep, steps, gpu_ops, union_busy, gaps

ev = prep(0)
st = steps(ev)

def fam(n):
    n = n.replace("void ", "")
    for tag in ("marlin_moe_wna16", "cross_device_reduce", "ncclDevKernel_AllGather",
                "ncclDevKernel_ReduceScatter", "ncclDevKernel_AllReduce", "sparse_fp8::flash_fwd",
                "flash_fwd_mla_combine", "deep_gemm::sm90_fp8_paged_mqa", "deep_gemm::sm90_fp8_mqa",
                "cooperative_topk", "act_and_mul", "moe_sum_vec", "moe_align_block_size",
                "count_and_sort_expert", "grouped_topk", "marlin::Marlin", "cutlass::device_kernel",
                "elementwise_kernel", "Memset", "Memcpy", "triton_", "nvjet_", "memcpy32_post",
                "StableTopK", "concat_and_cache", "indexer_k_quant", "deep_gemm::fp8_gemm"):
        if tag in n:
            return tag
    return n.split("(")[0][:40]

agg_idle = collections.Counter()
agg_cnt = collections.Counter()
agg_region = collections.Counter()
nsteps = 0
for si, (a, b) in enumerate(st):
    rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and a <= e["t"] < b]
    gl = sorted([e for e in rt if "GraphLaunch" in e["name"]], key=lambda e: e["t"])
    corr_target = gl[0]["args"]["correlation"]
    ops = gpu_ops(ev, a, b)
    tgt = [e for e in ops if e["args"].get("correlation") == corr_target]
    ts, te = min(e["t"] for e in tgt), max(e["t"] + e["dur"] / 1000 for e in tgt)

    ub, merged = union_busy(ops)
    gg = gaps(merged, a, b)
    nsteps += 1
    # successor kernel for each gap
    ops_sorted = sorted(ops, key=lambda e: e["t"])
    idx = 0
    for x, y in gg:
        while idx < len(ops_sorted) and ops_sorted[idx]["t"] < y - 1e-9:
            idx += 1
        succ = ops_sorted[idx]["name"] if idx < len(ops_sorted) else "END-OF-STEP"
        f = fam(succ)
        agg_idle[f] += (y - x)
        agg_cnt[f] += 1
        region = ("prologue" if y <= ts else "target-graph" if y <= te else "draft-tail")
        agg_region[region] += (y - x)

print(f"idle attribution over {nsteps} steps, rank0 (ms/step)")
print(f"{'successor kernel family':42s} {'idle/step':>10s} {'gaps/step':>10s} {'avg us':>9s}")
for f, v in agg_idle.most_common(22):
    print(f"  {f:40s} {v/nsteps:10.3f} {agg_cnt[f]/nsteps:10.1f} {1000*v/agg_cnt[f]:9.2f}")
print(f"\n  {'TOTAL':40s} {sum(agg_idle.values())/nsteps:10.3f} {sum(agg_cnt.values())/nsteps:10.1f}")
print("\nidle by region (ms/step):")
for r, v in agg_region.most_common():
    print(f"  {r:20s} {v/nsteps:8.3f}")
