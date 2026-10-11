"""Kernel groups per window (mean over the 4 devices, ms) for the prof_sizes.py
windows: nsys.<i>.sqlite in the given run dir, labels from timings.jsonl."""
import collections
import json
import sqlite3
import sys

GROUPS = [
    ("moe_gemm", "tiered_prefill::gemm"), ("moe_other", "tiered_prefill::"),
    ("q_allgather", "AllGather"), ("allreduce", "AllReduce"),
    ("attn", "sparse_attn_fwd"), ("attn_dec", "flash_fwd_splitkv_mla"),
    ("lse_combine", "lse_combine"), ("barrier", "barrier_kernel"),
    ("dense_gemm", "nvjet"), ("idx_logits", "mqa_logits"), ("idx_topk", "topKPerRow"),
    ("idx_other", "indexer"), ("kv_gather", "cp_gather"), ("drafter_moe", "marlin"),
    ("triton", "triton_"), ("elementwise", "elementwise"), ("cat/copy", "Cat"),
    ("fill", "Fill"),
]


def group(n):
    for g, s in GROUPS:
        if s in n:
            return g
    return "other"


run = sys.argv[1]
labels = {r["window"]: r for r in map(json.loads, open(f"{run}/timings.jsonl")) if r["profiled"]}
cols, spans = {}, {}
for w, r in sorted(labels.items()):
    c = sqlite3.connect(f"{run}/nsys.{w}.sqlite")
    names = dict(c.execute("select id, value from StringIds"))
    tot = collections.Counter()
    devs = set()
    sp = []
    for dev, s0, s1 in c.execute("select deviceId, min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId"):
        sp.append((s1 - s0) / 1e6)
    for dev, n, t in c.execute("select deviceId, demangledName, sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId, demangledName"):
        tot[group(names[n])] += t / 1e6
        devs.add(dev)
    key = f"{r['label']} ({r['prompt_tokens']})"
    cols[key] = {g: v / len(devs) for g, v in tot.items()}
    cols[key]["SUM kernels"] = sum(tot.values()) / len(devs)
    spans[key] = (sum(sp) / len(sp), r["ttft_s"] * 1e3)
keys = list(cols)
print("| group | " + " | ".join(keys) + " |")
print("| --- |" + " ---: |" * len(keys))
print("| GPU span / TTFT | " + " | ".join(f"{spans[k][0]:.0f} / {spans[k][1]:.0f}" for k in keys) + " |")
allg = sorted({g for k in keys for g in cols[k]}, key=lambda g: -cols[keys[-1]].get(g, 0))
for g in allg:
    print(f"| {g} | " + " | ".join(f"{cols[k].get(g, 0):.1f}" for k in keys) + " |")
