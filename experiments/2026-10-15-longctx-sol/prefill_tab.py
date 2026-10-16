"""Kernel groups (sum of kernel time, mean over the 4 devices, ms) of the
prof_longctx.py prefill windows: run dir nsys.<i>.sqlite, the prefill rows of
timings.jsonl in window order after the decode windows. Groups as
../2026-10-11-prefill-sizes/sizes_table.py.

    prefill_tab.py <run dir>
"""
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
    return next((g for g, s in GROUPS if s in n), "other")


run = sys.argv[1]
rows = [r for r in map(json.loads, open(f"{run}/timings.jsonl")) if r["what"] == "prefill"]
ndec = sum(1 for r in map(json.loads, open(f"{run}/timings.jsonl")) if r["what"] == "decode")
cols = {}
for w, r in enumerate(rows, ndec + 1):
    c = sqlite3.connect(f"{run}/nsys.{w}.sqlite")
    names = dict(c.execute("select id, value from StringIds"))
    tot = collections.Counter()
    for _, n, t in c.execute("select deviceId, demangledName, sum(end-start) from "
                             "CUPTI_ACTIVITY_KIND_KERNEL group by deviceId, demangledName"):
        tot[group(names[n])] += t / 4e6
    sp = [(b - a) / 1e6 for a, b in c.execute(
        "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId")]
    tot["(window span)"] = sum(sp) / len(sp)
    tot["(TTFT unprofiled)"] = r["ttft_unprofiled_s"] * 1e3
    cols[f"{r['label']} ({r['prompt_tokens'][1]})"] = tot
keys = sorted({k for t in cols.values() for k in t}, key=lambda k: -max(t[k] for t in cols.values()))
print("| ms | " + " | ".join(cols) + " |")
print("|---|" + "--:|" * len(cols))
for k in keys:
    if max(t[k] for t in cols.values()) >= 1:
        print(f"| {k} | " + " | ".join(f"{t[k]:.1f}" for t in cols.values()) + " |")
