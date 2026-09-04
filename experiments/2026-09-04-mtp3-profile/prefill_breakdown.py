#!/usr/bin/env python3
"""Complete kernel, roofline and communication breakdown of one prefill chunk.

Every GPU kernel is joined to the CPU operator that launched it through the
profiler's `External id`, so operand shapes come from `record_shapes` rather
than from a model of what the kernel ought to have been given.

Hot and cold expert tiers run the *same* Marlin kernel on the same stream and
their expert counts overlap (hot 26-49, cold 15-38 per rank, hot < cold in
39 of 75 layers), so neither the kernel name nor the count identifies a tier.
They are identified instead by matching each layer's observed pair against the
(hot, cold) counts the placement profile predicts for that layer, and the match
rate is reported so the identification can be checked rather than trusted.

FLOPs and logical bytes are derived from the joined shapes and the model
config. They are not measured DRAM transactions: caches, repeated experts and
compression metadata all move the physical number. Durations, launch counts and
shapes are observed. Ceilings are the project's conservative GH200 values,
carried from `2026-07-18-mtp-prompt-profile/roofline-analysis.md`.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path

HBM_BW, C2C_BW, NVLINK_PEER = 3.5e12, 421e9, 150e9
BF16_PEAK, FP8_PEAK = 630e12, 1260e12
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}

H, TOPK, NEXP, EP = 6144, 8, 256, 4
MOE_INT, HEADS, HEADS_LOCAL = 2048, 64, 16
KV_LORA, ROPE, VDIM = 512, 64, 512
IDX_HEADS, IDX_DIM, IDX_TOPK = 32, 128, 2048
ROUTED_LAYERS = 75


def load(path):
    with gzip.open(path, "rt") as fh:
        blob = json.load(fh)
    ev = [e for e in blob["traceEvents"] if e.get("ph") == "X"]
    o = min(e["ts"] for e in ev if e.get("cat") in GPU_CATS)
    for e in ev:
        e["t"] = (e["ts"] - o) / 1000
    return ev


def union_ms(evts):
    iv = sorted((e["t"], e["t"] + e["dur"] / 1000) for e in evts)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if ce is not None else 0.0)


def innermost(ext, cpu_by_ext):
    ops = [o for o in cpu_by_ext.get(ext, []) if o["cat"] == "cpu_op"]
    return min(ops, key=lambda o: o["dur"]) if ops else None


def expected_tiers(profile):
    d = json.load(open(profile))
    out = []
    for L in range(len(d["hot_experts"])):
        hs, ow = set(d["hot_experts"][L]), d["owners"][L]
        h = sum(1 for e in hs if ow[e] == 0)
        tot = sum(1 for e in range(NEXP) if ow[e] == 0)
        out.append((h, tot - h))
    return out


def marlin_kind(dims):
    """(gemm, experts) for a moe_wna16_marlin_gemm call, from its operands."""
    a, bq = dims[0], dims[2]
    return ("w13" if a[1] == H else "w2"), bq[0]


def analyse(path, tiers):
    ev = load(path)
    cpu_by_ext = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in ("cpu_op", "user_annotation"):
            cpu_by_ext[e["args"].get("External id")].append(e)
    by_corr = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in GPU_CATS and e.get("args", {}).get("correlation") is not None:
            by_corr[e["args"]["correlation"]].append(e)
    launches = sorted((e for e in ev if e.get("cat") in LAUNCH_CATS), key=lambda e: e["t"])
    anns = sorted((e for e in ev if e.get("cat") == "user_annotation"
                   and e["name"].startswith("execute_")), key=lambda e: e["t"])

    chunks = []
    for start, end in [(a["t"], b["t"]) for a, b in zip(anns, anns[1:])]:
        corrs = [e["args"]["correlation"] for e in launches
                 if start <= e["t"] < end and "correlation" in e.get("args", {})]
        ops = sorted((k for c in corrs for k in by_corr[c]), key=lambda e: e["t"])
        if not ops:
            continue
        meta = {}
        for k in ops:
            op = innermost(k["args"].get("External id"), cpu_by_ext)
            meta[id(k)] = op

        # Identify Marlin tiers by walking layers in launch order.
        marlin = [k for k in ops if "marlin_moe_wna16" in k["name"]]
        tier_of, matched, total = {}, 0, 0
        groups = [marlin[i:i + 4] for i in range(0, len(marlin), 4)]
        for li, g in enumerate(groups):
            if li >= len(tiers) or len(g) != 4:
                continue
            hot_n, cold_n = tiers[li]
            for k in g:
                op = meta.get(id(k))
                if op is None:
                    continue
                gemm, e_n = marlin_kind(op["args"]["Input Dims"])
                total += 1
                if e_n == hot_n and hot_n != cold_n:
                    tier_of[id(k)] = ("hot", gemm); matched += 1
                elif e_n == cold_n and hot_n != cold_n:
                    tier_of[id(k)] = ("cold", gemm); matched += 1
                else:
                    tier_of[id(k)] = ("ambiguous", gemm)

        rows = collections.defaultdict(
            lambda: {"n": 0, "us": 0.0, "dims": None, "types": None,
                     "kern": collections.Counter(), "calls": set()})
        for k in ops:
            op = meta.get(id(k))
            name = op["name"] if op else "(unjoined)"
            dims = op["args"].get("Input Dims") if op else None
            if id(k) in tier_of:
                tier, gemm = tier_of[id(k)]
                key = (name, f"{tier} {gemm}")
            elif name == "aten::mm" and dims:
                key = (name, f"{dims[0][1]}x{dims[1][1]}")
            else:
                key = (name, "")
            r = rows[key]
            r["n"] += 1
            r["us"] += k["dur"]
            r["kern"][k["name"]] += k["dur"]
            r["calls"].add(k["args"].get("External id"))
            if r["dims"] is None and dims:
                r["dims"] = dims
                r["types"] = op["args"].get("Input type")
        chunks.append({"rows": rows, "busy": union_ms(ops),
                       "wall": end - start, "match": (matched, total)})
    return chunks


def flops_bytes(op, tier, dims, calls, ctx_tokens):
    """(FLOPs, bytes, roof_name) per chunk for one role. None where not modeled."""
    M = 8192
    if op == "_moe_C::moe_wna16_marlin_gemm":
        E, kp, npk = dims[2][0], dims[2][1], dims[2][2]
        sc = dims[4]
        wb = (E * kp * npk * 4 + sc[0] * sc[1] * sc[2] * 2) * calls
        rows = M * TOPK / EP
        if tier.endswith("w13"):
            f = 2 * rows * H * 2 * MOE_INT * calls / ROUTED_LAYERS * (ROUTED_LAYERS / max(calls, 1))
            f = 2 * rows * H * (2 * MOE_INT)
        else:
            f = 2 * rows * MOE_INT * H
        f *= calls / max(1, calls)
        f = f * calls / calls if calls else 0
        return f * calls / max(calls, 1) * calls, wb, ("C2C" if tier.startswith("cold") else "HBM")
    if op == "aten::mm":
        a, b = dims[0], dims[1]
        f = 2 * a[0] * a[1] * b[1] * calls
        by = 2 * (a[0] * a[1] + a[1] * b[1] + a[0] * b[1]) * calls
        return f, by, "HBM"
    if op == "_flashmla_C::sparse_decode_fwd":
        S, D = dims[2][2], dims[0][3]
        f = 2 * M * HEADS * S * (D + VDIM) * calls
        by = M * S * (KV_LORA + ROPE + 80) * calls
        return f, by, "FP8"
    if op == "vllm::sparse_attn_indexer":
        f = 2 * M * IDX_HEADS * ctx_tokens * IDX_DIM * calls
        by = ctx_tokens * 132 * 64 * calls
        return f, by, "FP8"
    if op == "_C::top_k_per_row_prefill":
        by = (dims[0][0] * dims[0][1] * 4 + dims[3][0] * dims[3][1] * 4) * calls
        return None, by, "HBM"
    if op == "_C::silu_and_mul":
        by = (dims[0][0] * dims[0][1] + dims[1][0] * dims[1][1]) * 2 * calls
        return None, by, "HBM"
    if op == "vllm::all_reduce":
        S = dims[0][0] * dims[0][1] * 2
        return None, S * calls, "NVLINK"
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", default="/e/project1/profound/alint77/traces/mtp3-profile-1665068")
    ap.add_argument("--profile", default="agent_space/profiles/glm53-w4a16-2496.json")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    tiers = expected_tiers(args.profile)
    root = Path(args.trace_root) / "prefill"
    per_rank = {}
    for p in sorted(root.glob("*.trace.json.gz")):
        r = int(p.name.split("_rank")[1].split(".")[0])
        per_rank[r] = analyse(p, tiers)
        m, t = per_rank[r][0]["match"]
        print(f"rank {r}: {len(per_rank[r])} chunks, Marlin tier match {m}/{t}", flush=True)

    n = len(per_rank[0])
    agg = collections.defaultdict(lambda: {"n": 0.0, "us": 0.0, "dims": None,
                                           "types": None, "kern": collections.Counter(),
                                           "calls": 0.0})
    for c in per_rank[0]:
        for key, r in c["rows"].items():
            a = agg[key]
            a["n"] += r["n"] / n
            a["us"] += r["us"] / n
            a["calls"] += len(r["calls"]) / n
            a["kern"].update(r["kern"])
            if a["dims"] is None:
                a["dims"], a["types"] = r["dims"], r["types"]
    busy = statistics.fmean(c["busy"] for c in per_rank[0])
    wall = statistics.fmean(c["wall"] for c in per_rank[0])
    cum = sum(a["us"] for a in agg.values()) / 1000
    ctx = 3.5 * 8192   # mean preceding context over chunks 1-6

    print(f"\nrank0, mean of {n} chunks: wall {wall:.1f} ms | GPU busy(union) {busy:.1f} ms | "
          f"cumulative kernel {cum:.1f} ms | cum/busy {cum/busy:.3f} "
          f"({'single-stream, shares additive' if cum/busy < 1.02 else 'OVERLAPPED'})")

    out = []
    for key, a in sorted(agg.items(), key=lambda i: -i[1]["us"]):
        ms = a["us"] / 1000
        kern = a["kern"].most_common(1)[0][0] if a["kern"] else ""
        out.append({"op": key[0], "tier": key[1], "launches": a["n"],
                    "calls": a["calls"], "ms": ms, "pct": 100 * ms / busy,
                    "dims": a["dims"], "kernel": kern})
    args_json = {"wall_ms": wall, "busy_ms": busy, "cumulative_ms": cum,
                 "chunks": n, "roles": out}
    if args.json:
        args.json.write_text(json.dumps(args_json, indent=1) + "\n")

    print(f"\n{'operator':36s} {'tier/shape':12s} {'calls':>6s} {'kern':>6s} "
          f"{'ms':>9s} {'%':>6s}  operand dims")
    for r in out[:30]:
        d = json.dumps([x for x in (r["dims"] or []) if x][:3])[:44]
        print(f"{r['op'][:36]:36s} {r['tier'][:12]:12s} {r['calls']:6.0f} {r['launches']:6.0f} "
              f"{r['ms']:9.3f} {r['pct']:6.2f}  {d}")
    tail = sum(r["ms"] for r in out[30:])
    print(f"{'(' + str(max(0,len(out)-30)) + ' smaller roles)':36s} {'':12s} {'':6s} {'':6s} "
          f"{tail:9.3f} {100*tail/busy:6.2f}")
    print(f"{'TOTAL':36s} {'':12s} {'':6s} {'':6s} {cum:9.3f} {100*cum/busy:6.2f}")


if __name__ == "__main__":
    main()
