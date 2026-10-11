"""Kernel groups (sizes_table.py's) and big memcpys inside each prefill step
of an nsys report, mean over devices: one row per prefill step (75 MoE
layers), ms of summed kernel time per group, step wall, tokens.

    step_groups.py report.sqlite
"""
import collections
import sqlite3
import sys

# groups as in ../2026-10-11-prefill-sizes/sizes_table.py
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

c = sqlite3.connect(sys.argv[1])
names = dict(c.execute("select id, value from StringIds"))
devs = [d for (d,) in c.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL")]
res = []
for dev in devs:
    rows = c.execute("select start,end,demangledName,gridX from CUPTI_ACTIVITY_KIND_KERNEL "
                     "where deviceId=? order by start", (dev,)).fetchall()
    cps = c.execute("select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? "
                    "and bytes >= 1048576", (dev,)).fetchall()
    # a prefill step: from the first kernel after the previous decode/prefill step's
    # last kernel... approximated by [first route - 2 ms, last combine + 2 ms] per run
    steps = []
    for s, e, dn, gx in rows:
        n = names[dn]
        if "tiered_prefill::route" in n:
            if not steps or s - steps[-1][1] > 15e6:
                steps.append([s, e, 0])
        elif "tiered_prefill::combine" in n and steps:
            steps[-1][1] = e
            steps[-1][2] = gx
    for i, (s, e, tok) in enumerate(steps):
        lo = max(s - 3e6, steps[i - 1][1] if i else 0)
        g = collections.Counter()
        for ks, ke, dn, _ in rows:
            if ks >= lo and ke <= e + 1e6:
                g[group(names[dn])] += (ke - ks) / 1e6
        cp = sum(b for cs, ce, b in cps if cs >= lo and ce <= e + 1e6)
        res.append((dev, i, tok, (e - lo) / 1e6, g, cp / 2**30))
by = collections.defaultdict(list)
for dev, i, tok, wall, g, cp in res:
    by[i].append((tok, wall, g, cp))
keys = [k for k, _ in GROUPS] + ["other"]
print("| step | tokens | wall ms | copies GiB | " + " | ".join(keys) + " |")
print("|---|--:|--:|--:|" + "--:|" * len(keys))
for i, v in sorted(by.items()):
    m = lambda f: sum(f(x) for x in v) / len(v)  # noqa: E731
    print(f"| {i} | {v[0][0]} | {m(lambda x: x[1]):.0f} | {m(lambda x: x[3]):.1f} | "
          + " | ".join(f"{m(lambda x: x[2].get(k, 0)):.1f}" for k in keys) + " |")
