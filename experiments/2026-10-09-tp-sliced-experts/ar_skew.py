"""Per rank: median layer_kernel duration, the MoE all-reduce that follows it,
and the attention all-reduce (the other one-shot AR), and the absolute
layer_kernel end skew vs the earliest rank for matched calls.

    ar_skew.py <rank0.gz> <rank1.gz> ...
"""
import gzip
import json
import statistics as st
import sys

per = []
for f in sys.argv[1:]:
    ks = sorted((e for e in json.load(gzip.open(f))["traceEvents"] if e.get("cat") == "kernel"),
                key=lambda e: e["ts"])
    moe, moe_ar, attn_ar, ends = [], [], [], []
    last = None
    for k in ks:
        n = k["name"]
        if "layer_kernel" in n or "gemm_kernel<1" in n:
            last = "moe"
            moe.append(k["dur"])
            ends.append(k["ts"] + k["dur"])
        elif "allreduce_fusion" in n:
            (moe_ar if last == "moe" else attn_ar).append(k["dur"])
            last = None
    per.append(ends)
    print(f"{f.split('/')[-1][:30]}: MoE kernel {st.median(moe):6.1f} us, MoE AR {st.median(moe_ar):5.1f}, "
          f"attn AR {st.median(attn_ar):5.1f}  (n={len(moe)})")
n = min(map(len, per))
lag = [[e[i] - min(p[i] for p in per) for i in range(n)] for e in per]
print("median lag of the MoE kernel end behind the earliest rank (us):",
      [round(st.median(l), 1) for l in lag], " mean max lag", round(st.mean(max(l[i] for l in lag) for i in range(n)), 1))
