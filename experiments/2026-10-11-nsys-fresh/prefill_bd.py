"""Wall-time breakdown of prefill steps from an nsys report: each instant of
the step goes to the highest-priority category with a kernel running (so PDL
overlap and side streams are not double-counted), none running = idle.
Steps are every 75 tiered_prefill MoE layers; a step spans
from the end of the previous step / decode step to its last MoE combine plus
the layer tail. Mean over devices.

    prefill_bd.py report.sqlite [report.sqlite ...]
"""
import collections
import re
import sqlite3
import sys

CATS = [  # priority order
    ("MoE GEMM", r"tiered_prefill::gemm"),
    ("MoE route/gather/act/combine", r"tiered_prefill::"),
    ("attention (sparse fwd)", r"sparse_attn_fwd|flash_fwd"),
    ("DCP q all-gather (NCCL)", r"AllGather"),
    ("DCP lse_combine", r"lse_combine|barrier_kernel"),
    ("TP all-reduce", r"AllReduce|cross_device_reduce|allreduce_fusion"),
    ("dense GEMM", r"nvjet|decode_gemm|cutlass_kernel_vllmmodel_executorkernelsattentiondsa(?!dcp_indexer)|gemm"),
    ("indexer", r"indexer|mqa_logits|topKPerRow|_pack_dcp_topk|_convert_req_index|cp_gather"),
    ("MTP drafter MoE (marlin)", r"[Mm]arlin"),
    ("router top-k", r"grouped_topk|moe_sum"),
    ("glue (triton, elementwise, norm, KV write)", r"."),
]
PAT = [(k, re.compile(p)) for k, p in CATS]


def cat(n):
    for k, p in PAT:
        if p.search(n):
            return k


for path in sys.argv[1:]:
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    devs = [d for (d,) in c.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL")]
    per = collections.defaultdict(list)
    for dev in devs:
        rows = [(s, e, names[dn], gx) for s, e, dn, gx in c.execute(
            "select start,end,demangledName,gridX from CUPTI_ACTIVITY_KIND_KERNEL "
            "where deviceId=? order by start", (dev,))]
        # step boundaries: every 75 MoE layers (one route kernel per layer)
        steps, nroute = [], 0
        for s, e, n, gx in rows:
            if "tiered_prefill::route" in n:
                if nroute % 75 == 0:
                    steps.append([s, e, 0])
                nroute += 1
            elif "tiered_prefill::combine" in n and steps:
                steps[-1][1], steps[-1][2] = e, gx
        bounds = []
        for i, (s, e, tok) in enumerate(steps):
            # start: first kernel after the previous step's tail / decode step;
            # take the first kernel within 15 ms before the first route
            lo = next(ks for ks, ke, n, _ in rows if ks >= s - 15e6)
            if i:
                lo = max(lo, bounds[-1][1])
                lo = next(ks for ks, ke, n, _ in rows if ks >= lo)
            # end: last kernel before the next 5 ms gap after the last combine
            hi = e
            for ks, ke, n, _ in rows:
                if ks < e:
                    continue
                if ks - hi > 5e6 or "tiered_decode" in n or (
                        i + 1 < len(steps) and ks >= steps[i + 1][0] - 15e6):
                    break
                hi = max(hi, ke)
            bounds.append((lo, hi, tok))
        for i, (lo, hi, tok) in enumerate(bounds):
            ev = []
            for ks, ke, n, _ in rows:
                if ke > lo and ks < hi:
                    ev.append((max(ks, lo), 0, cat(n)))
                    ev.append((min(ke, hi), 1, cat(n)))
            ev.sort()
            active = collections.Counter()
            acc = collections.Counter()
            t = lo
            order = [k for k, _ in CATS]
            for ts, kind, k in ev:
                top = next((x for x in order if active[x] > 0), "idle")
                acc[top] += ts - t
                t = ts
                active[k] += 1 if kind == 0 else -1
            acc["idle"] += hi - t
            per[i].append((tok, (hi - lo) / 1e6, {k: v / 1e6 for k, v in acc.items()}))
    print(f"## {path}")
    for i, v in sorted(per.items()):
        tok, wall = v[0][0], sum(x[1] for x in v) / len(v)
        m = collections.Counter()
        for _, _, d in v:
            for k, x in d.items():
                m[k] += x / len(v)
        print(f"step {i}: {tok} tokens, wall {wall:.1f} ms")
        for k in [k for k, _ in CATS] + ["idle"]:
            if m[k] >= 0.05:
                print(f"   {m[k]:7.1f} ms {100*m[k]/wall:5.1f}%  {k}")
