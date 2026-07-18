"""Pass 4: per-kernel-family hardware analysis (occupancy, grids, duration mix)."""

import json
import collections
import statistics
from analyze import classify
from analyze3 import load

FAMS = [
    "routed_moe_marlin", "custom_allreduce", "dense_w4_machete", "dense_w4_marlin",
    "sparse_mla", "dsa_scan", "dsa_topk", "moe_support", "elementwise_meta",
    "mla_bf16_gemm", "vocab_gemm", "router", "router_gemm", "mtp_fp8_gemm",
]


def pct(d, q):
    d = sorted(d)
    return d[min(len(d) - 1, int(q * len(d)))]


def main():
    rank = 0
    kernels, windows = load(rank)
    lo = windows[1][0]
    hi = windows[-1][0]
    n_steps = len(windows) - 2
    ks = [e for e in kernels if lo <= e["ts"] < hi]
    fam = collections.defaultdict(list)
    for e in ks:
        comp, _ = classify(e["name"])
        fam[comp].append(e)

    print(f"rank {rank}, {n_steps} steps, {len(ks)} kernels")
    print(f"{'family':22s} {'n/st':>5} {'mean':>6} {'p50':>6} {'p90':>6} {'p99':>6} "
          f"{'occ%':>5} {'blk/SM':>7} {'grids'}")
    for f in FAMS:
        es = fam.get(f, [])
        if not es:
            continue
        d = [e["dur"] for e in es]
        occ = [e["args"].get("est. achieved occupancy %", -1) for e in es]
        bsm = [e["args"].get("blocks per SM", -1) for e in es]
        grids = collections.Counter(tuple(e["args"].get("grid", [])) for e in es)
        gtop = " ".join(f"{g}x{c//n_steps}" for g, c in grids.most_common(3))
        print(f"{f:22s} {len(es)/n_steps:5.0f} {statistics.mean(d):6.1f} "
              f"{pct(d,0.5):6.1f} {pct(d,0.9):6.1f} {pct(d,0.99):6.1f} "
              f"{statistics.mean(occ):5.0f} {statistics.mean(bsm):7.2f} {gtop}")

    # routed marlin: cluster by duration to separate populations
    md = sorted(e["dur"] for e in fam["routed_moe_marlin"])
    print("\nrouted marlin duration deciles (us):",
          [round(pct(md, q / 10), 1) for q in range(10)] + [round(md[-1], 1)])
    # split calls into short/long halves by per-layer position: pair structure
    marlins = sorted(fam["routed_moe_marlin"], key=lambda e: e["ts"])
    # group into layers of 4 consecutive calls (hot13, hot2, cold13, cold2 across streams)
    per4 = collections.defaultdict(list)
    for i, e in enumerate(marlins):
        per4[i % 4].append(e["dur"])
    # AR jitter
    ar = [e["dur"] for e in fam["custom_allreduce"]]
    print(f"\ncustom AR per-call us: mean {statistics.mean(ar):.1f} p50 {pct(ar,0.5):.1f} "
          f"p90 {pct(ar,0.9):.1f} p99 {pct(ar,0.99):.1f} max {max(ar):.1f} "
          f"(isolated floor 4.06 us)")
    # machete grid diversity = layer shapes
    mach = fam["dense_w4_machete"]
    byg = collections.defaultdict(list)
    for e in mach:
        byg[tuple(e["args"]["grid"])].append(e["dur"])
    print("\nmachete populations (grid: n/step, mean us):")
    for g, ds in sorted(byg.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {str(g):18s} n={len(ds)/n_steps:5.1f} mean={statistics.mean(ds):6.1f} "
              f"sum/step={sum(ds)/n_steps:7.1f}")
    # sparse MLA splitkv grids
    for name, key in (("sparse_mla", "flash_fwd_splitkv"), ("dsa_scan", "mqa_logits"),
                      ("dsa_topk", "cooperative_topk")):
        es = [e for e in fam[name] if key in e["name"]]
        if not es:
            continue
        byg = collections.Counter(tuple(e["args"]["grid"]) for e in es)
        d = [e["dur"] for e in es]
        occ = statistics.mean(e["args"].get("est. achieved occupancy %", 0) for e in es)
        print(f"\n{key}: n/step={len(es)/n_steps:.0f} mean={statistics.mean(d):.1f}us "
              f"occ={occ:.0f}% grids={dict(list(byg.items())[:3])}")


if __name__ == "__main__":
    main()
