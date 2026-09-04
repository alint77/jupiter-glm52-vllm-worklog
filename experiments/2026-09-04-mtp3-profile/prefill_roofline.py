#!/usr/bin/env python3
"""Prefill chunk: kernel inventory, roofline, layer sequence, comms, rank delta.

Kernels are joined to their launching CPU operator through the profiler's
`External id`, so operand shapes are observed, not modelled.

Hot and cold expert tiers run the same Marlin kernel on the same stream and
their expert counts overlap, so neither name nor count identifies them. They
are separated by launch order, validated against effective weight bandwidth:
across 3648 pairs the first call of each layer's pair is the faster one in
98.16% of cases, and the two populations do not overlap (hot 240-270 GB/s,
cold 81-123 GB/s). Every pair sums to the rank's 64 owned experts.

FLOPs and logical bytes come from those shapes plus the model config; they are
not measured DRAM transactions. Routed rows per tier are not observable from
shapes, so they are apportioned by expert count -- which understates hot and
overstates cold, since hot experts are hot precisely because they take more
tokens. Ceilings are the project's conservative GH200 values from
2026-07-18-mtp-prompt-profile/roofline-analysis.md.
"""
from __future__ import annotations
import argparse, collections, gzip, json, statistics
from pathlib import Path

HBM, C2C, NVL_PEER = 3.5e12, 421e9, 150e9
BF16, FP8 = 630e12, 1260e12
GPU = {"kernel", "gpu_memcpy", "gpu_memset"}
LAU = {"cuda_runtime", "cuda_driver"}
Hd, TOPK, EP, NEXP = 6144, 8, 4, 256
MOE_I, HEADS, HLOCAL, VDIM = 2048, 64, 16, 512
IDX_H, IDX_D = 32, 128


def load(p):
    with gzip.open(p, "rt") as fh:
        ev = [e for e in json.load(fh)["traceEvents"] if e.get("ph") == "X"]
    o = min(e["ts"] for e in ev if e.get("cat") in GPU)
    for e in ev:
        e["t"] = (e["ts"] - o) / 1000
    return ev


def chunks_of(p):
    ev = load(p)
    cpu = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") == "cpu_op":
            cpu[e["args"].get("External id")].append(e)
    bc = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in GPU and e.get("args", {}).get("correlation") is not None:
            bc[e["args"]["correlation"]].append(e)
    lau = sorted((e for e in ev if e.get("cat") in LAU), key=lambda e: e["t"])
    ann = sorted((e for e in ev if e.get("cat") == "user_annotation"
                  and e["name"].startswith("execute_")), key=lambda e: e["t"])
    for s, en in [(a["t"], b["t"]) for a, b in zip(ann, ann[1:])]:
        cs = [e["args"]["correlation"] for e in lau
              if s <= e["t"] < en and "correlation" in e.get("args", {})]
        ops = sorted((k for c in cs for k in bc[c]), key=lambda e: e["t"])
        if not ops:
            continue
        for k in ops:
            l = [q for q in cpu.get(k["args"].get("External id"), []) if q["cat"] == "cpu_op"]
            k["op"] = min(l, key=lambda q: q["dur"]) if l else None
        yield ops, en - s


def role_of(k, tier):
    op = k.get("op")
    if op is None:
        return ("(unjoined)", "")
    n, d = op["name"], op["args"].get("Input Dims")
    if n == "_moe_C::moe_wna16_marlin_gemm" and d:
        return (n, f"{tier.get(id(k),'?')} {'w13' if d[0][1] == Hd else 'w2'}")
    if n == "aten::mm" and d:
        return (n, f"{d[0][1]}x{d[1][1]}")
    return (n, "")


def tag_tiers(ops):
    """Hot = first of each layer's same-GEMM pair; validated by bandwidth."""
    tier = {}
    for gemm in ("w13", "w2"):
        seq = [k for k in ops if "marlin_moe_wna16" in k["name"] and k.get("op")
               and (k["op"]["args"]["Input Dims"][0][1] == Hd) == (gemm == "w13")]
        for i in range(0, len(seq) - 1, 2):
            a, b = seq[i], seq[i + 1]
            da = a["op"]["args"]["Input Dims"][2][0]
            db = b["op"]["args"]["Input Dims"][2][0]
            if da + db != 64:
                tier[id(a)] = tier[id(b)] = "single"
                continue
            tier[id(a)], tier[id(b)] = "hot", "cold"
    return tier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/e/project1/profound/alint77/traces/mtp3-profile-1665068/prefill")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    root = Path(a.root)

    per_rank, seq_dump = {}, None
    for p in sorted(root.glob("*.trace.json.gz")):
        rank = int(p.name.split("_rank")[1].split(".")[0])
        agg = collections.defaultdict(lambda: {"k": 0, "us": 0.0, "calls": set(),
                                               "dims": None, "kern": collections.Counter()})
        walls, busies, nch = [], [], 0
        per_chunk_role = []
        for ops, wall in chunks_of(p):
            nch += 1
            tier = tag_tiers(ops)
            walls.append(wall)
            iv = sorted((k["t"], k["t"] + k["dur"] / 1000) for k in ops)
            tot, cs, ce = 0.0, None, None
            for s0, e0 in iv:
                if ce is None or s0 > ce:
                    if ce is not None:
                        tot += ce - cs
                    cs, ce = s0, e0
                else:
                    ce = max(ce, e0)
            busies.append(tot + (ce - cs))
            this = collections.Counter()
            for k in ops:
                r = role_of(k, tier)
                x = agg[r]
                x["k"] += 1
                x["us"] += k["dur"]
                x["calls"].add(k["args"].get("External id"))
                x["kern"][k["name"]] += k["dur"]
                this[r] += k["dur"]
                if x["dims"] is None and k.get("op"):
                    x["dims"] = k["op"]["args"].get("Input Dims")
            per_chunk_role.append(this)
            if rank == 0 and nch == 3 and seq_dump is None:
                seq_dump = (ops, tier)
        per_rank[rank] = {"agg": agg, "n": nch, "wall": statistics.fmean(walls),
                          "busy": statistics.fmean(busies), "per_chunk": per_chunk_role}
        print(f"rank {rank}: {nch} chunks", flush=True)

    A = per_rank[0]
    n, busy, wall = A["n"], A["busy"], A["wall"]
    rows = []
    for key, v in A["agg"].items():
        rows.append({"op": key[0], "tag": key[1], "ms": v["us"] / 1000 / n,
                     "kern": v["k"] / n, "calls": len(v["calls"]) / n,
                     "dims": v["dims"],
                     "kernel": v["kern"].most_common(1)[0][0] if v["kern"] else ""})
    rows.sort(key=lambda r: -r["ms"])
    cum = sum(r["ms"] for r in rows)

    print(f"\n{'='*104}\nPREFILL CHUNK (8192 tokens), rank 0, mean of {n} chunks")
    print(f"wall {wall:.1f} ms | GPU busy(union) {busy:.1f} ms | cumulative kernel {cum:.1f} ms"
          f" | cum/busy {cum/busy:.3f} -> single-stream, shares are additive\n{'='*104}")
    print(f"{'operator':34s} {'tier/shape':11s} {'calls':>6s} {'ms':>9s} {'%':>6s}  {'operands'}")
    for r in rows[:26]:
        d = json.dumps([x for x in (r["dims"] or []) if x][:3])[:42]
        print(f"{r['op'][:34]:34s} {r['tag'][:11]:11s} {r['calls']:6.0f} {r['ms']:9.3f} "
              f"{100*r['ms']/busy:6.2f}  {d}")
    tail = sum(r["ms"] for r in rows[26:])
    print(f"{'(' + str(len(rows)-26) + ' smaller roles)':34s} {'':11s} {'':6s} {tail:9.3f} {100*tail/busy:6.2f}")
    print(f"{'TOTAL':34s} {'':11s} {'':6s} {cum:9.3f} {100*cum/busy:6.2f}")

    # ---------------- per-chunk trend ----------------
    print(f"\n{'-'*104}\nPER-CHUNK TREND (does it grow with context?)  ms per chunk, rank 0")
    watch = [("_flashmla_C::sparse_decode_fwd", ""), ("vllm::sparse_attn_indexer", ""),
             ("_C::top_k_per_row_prefill", ""), ("vllm::all_reduce", ""),
             ("_moe_C::moe_wna16_marlin_gemm", "hot w13")]
    print(f"{'role':44s} " + " ".join(f"{'c'+str(i+1):>9s}" for i in range(n)))
    for key in watch:
        vals = [c.get(key, 0.0) / 1000 for c in A["per_chunk"]]
        print(f"{(key[0]+' '+key[1])[:44]:44s} " + " ".join(f"{v:9.2f}" for v in vals))

    if a.out:
        a.out.write_text(json.dumps({"wall": wall, "busy": busy, "cum": cum, "n": n,
                                     "roles": rows}, indent=1, default=str) + "\n")
        print(f"\nwrote {a.out}")

    # ---------------- rank delta ----------------
    print(f"\n{'-'*104}\nRANK 1 MINUS MEAN(OTHERS), ms per chunk  (rank 1 arrives last 58.8% of the time)")
    keys = [(r["op"], r["tag"]) for r in rows[:14]]
    print(f"{'role':46s} {'r0':>8s} {'r1':>8s} {'r2':>8s} {'r3':>8s} {'r1-mean':>9s}")
    for key in keys:
        vs = []
        for rk in sorted(per_rank):
            v = per_rank[rk]["agg"].get(key)
            vs.append((v["us"] / 1000 / per_rank[rk]["n"]) if v else 0.0)
        others = statistics.fmean([vs[0], vs[2], vs[3]])
        print(f"{(key[0]+' '+key[1])[:46]:46s} " + " ".join(f"{v:8.2f}" for v in vs)
              + f" {vs[1]-others:+9.2f}")
    print(f"{'-- GPU busy per chunk --':46s} "
          + " ".join(f"{per_rank[r]['busy']:8.2f}" for r in sorted(per_rank)))

    # ---------------- sequence ----------------
    ops, tier = seq_dump
    ar = [i for i, k in enumerate(ops) if "ncclDevKernel_AllReduce" in k["name"]]
    lo, hi = ar[40], ar[42]
    print(f"\n{'-'*104}\nONE MoE LAYER IN LAUNCH ORDER (chunk 3, between all-reduce #40 and #42)")
    print(f"{'#':>3s} {'us':>9s}  {'kernel':62s} {'operator'}")
    for j, k in enumerate(ops[lo:hi + 1]):
        op = (k.get("op") or {}).get("name", "")
        t = tier.get(id(k), "")
        print(f"{j:3d} {k['dur']:9.1f}  {k['name'][:62]:62s} {op[:34]}{(' ['+t+']') if t else ''}")


if __name__ == "__main__":
    main()
