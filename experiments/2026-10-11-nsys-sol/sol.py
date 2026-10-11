"""Per-kernel SOL table across concurrencies from the nsys reports
(graph kernels per node): calls and us per call inside the verify graph, mean
over the 4 ranks and the per-call minimum over ranks, against a roofline per
call. PDL dependents (finalize, FlashMLA combine) are charged only their tail
past the producer's end (memory: PDL kernel time is not cost).

    sol.py <window: 1 = ~5K, 2 = ~50K> [--md]
"""
import collections
import json
import os
import sqlite3
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/e/fscratch/profound/naeimitabiei1/sliced-prof")
RUNS = {1: "nsol-c1-2272145", 2: "nsol-c2-2272143", 4: "nsol-c4-2272144",
        8: "nsol-c8-2272142", 16: "nsol-c16-2272141"}
# SOL_RUNS="1:<run>,2:<run>,..." and SOL_AN=<dir> analyse another set of runs
if os.environ.get("SOL_RUNS"):
    RUNS = {int(k): v for k, v in (kv.split(":") for kv in os.environ["SOL_RUNS"].split(","))}
AN = Path(os.environ.get("SOL_AN", Path(__file__).parent / "an"))
HBM, C2C, NVL, BF16 = 3.6e12, 0.419e12, 0.43e12, 630e12
# MoE layer_kernel floor per call, us: max(hot/HBM, cold/C2C) at the mean
# routing mix (../2026-10-09-tp-sliced-experts README); c1 = DFlash2 8 tokens
# 38 hot / 4 cold; c2 interpolated between n=1 and n=4
MOE_FLOOR = {1: 61.3, 2: 61.0, 4: 119.0, 8: 221.0, 16: 378.0}
# dense weight bytes per step (decode-dive-3 shapes): o_proj, fused qkv_a,
# q_b + indexer wq_b (99 calls), router, W_UV, W_UK
DENSE_BYTES = 78 * 50.3e6 + 78 * 32.5e6 + 99 * 16.8e6 + 75 * 3.1e6 + 78 * 4.2e6 + 78 * 3.1e6
GROUPS = [
    ("MoE layer_kernel", ("tiered_decode::layer_kernel",)),
    ("MoE finalize (tail)", ("tiered_decode::finalize_kernel",)),
    ("MoE route_prep + topk", ("route_prep_kernel", "grouped_topk_fused")),
    ("dense GEMMs", ("decode_gemm_kernel", "nvjet_sm90", "cute_dsl_ll_bf16", "splitKreduce",
                     "triton_tem_fused_mm")),
    ("FlashMLA sparse", ("flash_fwd_splitkv_mla_fp8_sparse",)),
    ("FlashMLA combine (tail)", ("flash_fwd_mla_combine",)),
    ("DCP gather_cat", ("one_shot::gather_cat",)),
    ("DCP lse_reduce_scatter", ("lse_reduce_scatter",)),
    ("AR + RMSNorm", ("allreduce_fusion_kernel",)),
    ("indexer (gather, logits, top-k, quant)", ("one_shot::gather_kernel", "paged_mqa_logits",
                                                "dcp_indexer", "filtered_topk", "cooperative_topk",
                                                "fused_indexer_q_rope", "indexer_k_quant",
                                                "pack_dcp_topk", "convert_req_index",
                                                "_dcp_decode_sched")),
    ("KV write", ("concat_and_cache_ds_mla",)),
    ("skip-KV staging (side stream)", ("_gather_rows_kernel", "_mark_rows", "_compact_rows",
                                       "_remap_rows")),
]
TAIL = {"MoE finalize (tail)": "tiered_decode::layer_kernel",
        "FlashMLA combine (tail)": "flash_fwd_splitkv_mla_fp8_sparse"}


def group_of(name):
    for g, pats in GROUPS:
        if any(p in name for p in pats):
            return g
    return "other verify-graph kernels"


def per_device(path, dev):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    by = collections.defaultdict(list)
    for s, e, corr, n in c.execute("select start, end, correlationId, demangledName from "
                                   "CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ?", (dev,)):
        by[corr].append((s, e, names[n]))
    steps = [sorted(v) for v in by.values() if len(v) >= 500]
    calls = collections.defaultdict(list)  # group -> per-call us (step-major order)
    for ks in steps:
        last_prod = {}
        for s, e, n in ks:
            g = group_of(n)
            if g in TAIL:
                pe = last_prod.get(TAIL[g])
                calls[g].append(max(0, e - max(s, pe if pe else s)) / 1e3)
                continue
            for tg, prod in TAIL.items():
                if prod in n:
                    last_prod[prod] = e
            calls[g].append((e - s) / 1e3)
    return len(steps), calls


def floors(conc):
    t = 8 if conc == 1 else 4 * conc
    fl = t * 64 * 768 * 1088 * 2 / BF16
    kv = t * 768 * 656 / HBM
    return {
        "MoE layer_kernel": (MOE_FLOOR[conc], "max(hot/HBM, cold/C2C)"),
        "dense GEMMs": (None, "weights 8.92 GB/step / HBM"),
        "FlashMLA sparse": (max(fl, kv) * 1e6, f"{t} tok x 64 h x 768 keys"),
        "DCP gather_cat": (3.5e6 * t / 64 / NVL * 1e6, "3.5 MB x T/64 over NVLink"),
        "DCP lse_reduce_scatter": (3.1e6 * t / 64 / NVL * 1e6, "3.1 MB x T/64 over NVLink"),
        "AR + RMSNorm": (2.36e6 * t / 64 / NVL * 1e6, "2.36 MB x T/64 over NVLink"),
    }


def main():
    w = sys.argv[1]
    out = {}
    for conc, run in RUNS.items():
        devs = [per_device(ROOT / run / f"nsys.{w}.sqlite", d) for d in range(4)]
        nsteps = [d[0] for d in devs]
        res = {}
        for g in [x for x, _ in GROUPS] + ["other verify-graph kernels"]:
            per = [d[1].get(g, []) for d in devs]
            if not all(per):
                continue
            n = min(len(p) for p in per)
            mins = [min(p[i] for p in per) for i in range(n)]
            ms_step = st.mean(sum(p) / k for p, k in zip(per, nsteps)) / 1e3
            res[g] = {"calls": st.mean(len(p) / k for p, k in zip(per, nsteps)),
                      "us": st.mean(st.mean(p) for p in per), "min_us": st.mean(mins),
                      "ms_step": ms_step}
        bd = [json.loads((AN / f"bd-c{conc}-w{w}-d{d}.json").read_text())
              for d in range(4)]
        res["_period"] = st.mean(b["period_ms"] for b in bd)
        res["_outside"] = st.mean(b["ms"]["outside the verify graph (draft, lm_head, sampling, prep)"]
                                  for b in bd)
        res["_idle"] = st.mean(b["ms"]["GPU idle"] for b in bd)
        out[conc] = res
    print(f"window {w}: per concurrency, us/call (min over ranks) and ms/step; SOL us/call; gap ms/step")
    for conc, res in out.items():
        fl = floors(conc)
        print(f"\n### c={conc}: period {res['_period']:.2f} ms, outside the verify graph "
              f"{res['_outside']:.2f}, idle {res['_idle']:.2f}")
        print("| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |")
        print("|---|--:|--:|--:|--:|--:|--:|---|")
        rows = []
        for g, r in res.items():
            if g.startswith("_"):
                continue
            sol, model = fl.get(g, (None, ""))
            if g == "dense GEMMs":
                sol_ms = DENSE_BYTES / HBM * 1e3
                gap = r["ms_step"] - sol_ms
                rows.append((gap, f"| {g} | {r['calls']:.0f} | {r['us']:.1f} | {r['ms_step']:.2f} | "
                                  f"({sol_ms:.2f} ms) | {100 * sol_ms / r['ms_step']:.0f} | {gap:.2f} | {model} |"))
                continue
            if sol:
                gap = r["ms_step"] - sol * r["calls"] / 1e3
                pct = f"{100 * sol / r['us']:.0f}"
                sols = f"{sol:.1f}"
            else:
                gap, pct, sols = float("nan"), "", ""
            rows.append((gap if gap == gap else -1, f"| {g} | {r['calls']:.0f} | {r['us']:.1f} ({r['min_us']:.1f}) | "
                                                    f"{r['ms_step']:.2f} | {sols} | {pct} | "
                                                    f"{'' if gap != gap else f'{gap:.2f}'} | {model} |"))
        for _, line in sorted(rows, key=lambda x: -x[0]):
            print(line)
    (AN / f"sol-w{w}.json").write_text(json.dumps(out, indent=1))


main()
