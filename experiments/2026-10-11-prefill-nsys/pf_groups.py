"""Per-device kernel groups for each prefill window: ms per window, and
NCCL / combine time on each rank vs the fastest rank (skew vs transfer)."""
import collections, sqlite3, sys

GROUPS = [
    ("moe_gemm", "tiered_prefill::gemm"), ("moe_gather", "tiered_prefill::gather"),
    ("moe_combine", "tiered_prefill::combine"), ("allgather", "AllGather"),
    ("allreduce", "AllReduce"), ("attn_prefill", "sparse_attn_fwd"),
    ("attn_decode", "flash_fwd_splitkv_mla"), ("lse_combine", "lse_combine"),
    ("barrier", "barrier_kernel"), ("dense_gemm", "nvjet"), ("act", "act_kernel"),
    ("indexer", "dcp_indexer"), ("mqa_logits", "mqa_logits"), ("topk", "topKPerRow"),
    ("decode_moe", "tiered_decode"), ("marlin", "marlin"), ("kv_gather", "cp_gather"),
    ("elementwise", "elementwise"),
]

def group(n):
    for g, s in GROUPS:
        if s in n:
            return g
    return "other"

for path in sys.argv[1:]:
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    per = collections.defaultdict(lambda: collections.Counter())
    span = {}
    for dev, s0, s1 in c.execute("select deviceId, min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId"):
        span[dev] = (s1 - s0) / 1e6
    for dev, n, t in c.execute("select deviceId, demangledName, sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId, demangledName"):
        per[dev][group(names[n])] += t / 1e6
    devs = sorted(per)
    print(f"== {path}  span " + " ".join(f"{span[d]:.0f}" for d in devs))
    tot = collections.Counter()
    for d in devs: tot.update(per[d])
    for g, _ in tot.most_common():
        v = [per[d][g] for d in devs]
        print(f"  {g:13s} " + " ".join(f"{x:7.1f}" for x in v) + f"   min {min(v):7.1f}  mean-min {sum(v)/len(v)-min(v):6.1f}")
