import sqlite3, collections, statistics as st, sys, re
c = sqlite3.connect(sys.argv[1]); names = dict(c.execute("select id, value from StringIds"))
pat = {"gather_cat": "gather_cat_kernel", "lse_rs": "lse_reduce_scatter_kernel", "gather": "one_shot::gather_kernel",
       "AR": "allreduce_fusion_kernel", "flashmla": "flash_fwd_splitkv_mla_fp8_sparse", "gather_rows": "_gather_rows_kernel",
       "layer_kernel": "layer_kernel"}
d = collections.defaultdict(lambda: collections.defaultdict(list))
for dev, s, e, n, corr in c.execute("select deviceId, start, end, demangledName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
    nm = names[n]
    for k, p in pat.items():
        if p in nm:
            d[k][dev].append((e - s) / 1e3)
for k in pat:
    devs = sorted(d[k]); L = min(len(d[k][x]) for x in devs)
    mins = [min(d[k][x][i] for x in devs) for i in range(L)]
    means = [st.mean(d[k][x][i] for x in devs) for i in range(L)]
    print(f"{k:12s} calls/dev {L:5d}  mean {st.mean(means):7.2f}  min-over-ranks {st.mean(mins):7.2f}  median-min {st.median(mins):7.2f} us")
