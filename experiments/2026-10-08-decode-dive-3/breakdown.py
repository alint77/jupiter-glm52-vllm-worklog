"""Decode step breakdown from one rank's torch-profiler trace: every instant
of a step (FlashMLA of layer 0 to the next step's) goes to the highest-
priority category among the kernels running then (so side-stream work hidden
under the main stream does not count twice); idle is time with no kernel.

    breakdown.py <trace.json.gz>
"""
import gzip
import json
import statistics as st
import sys

MLA = "flash_fwd_splitkv_mla_fp8_sparse"
# (category, substrings), highest priority first
CATS = [
    ("MoE GEMMs (tiered one-kernel w13/w2)", ["tiered_decode::gemm_kernel"]),
    ("MoE route/act/finalize + router", ["tiered_decode", "grouped_topk", "marlin_moe"]),
    ("attention (FlashMLA main)", [MLA]),
    ("attention (split-KV combine)", ["flash_fwd_mla_combine"]),
    ("all-reduce + RMSNorm (fused, incl. wait)", ["allreduce_fusion", "cross_device_reduce"]),
    ("DCP collectives (gather / LSE reduce-scatter)", ["one_shot::"]),
    ("DSA indexer", ["mqa_logits", "cooperative_topk", "StableTopK", "indexer",
                     "_pack_dcp_topk", "convert_req_index"]),
    ("dense GEMMs (decode_gemm)", ["decode_gemm_kernel"]),
    ("dense GEMMs (cuBLAS / CUTLASS / cute-dsl)", ["nvjet", "splitKreduce", "cutlass",
                                                   "deep_gemm", "dotprod", "splitk"]),
    ("KV write / skip-KV staging", ["concat_and_cache", "reshape_and_cache", "_gather_rows",
                                    "_mark_rows", "_compact_rows", "_remap_rows"]),
    ("drafter / sampling / other", [""]),
]


def cat(name: str) -> int:
    for i, (_, keys) in enumerate(CATS):
        if any(k in name for k in keys):
            return i
    return len(CATS) - 1


def main():
    ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
    ks = sorted(((e["ts"], e["ts"] + e["dur"], cat(e["name"])) for e in ev
                 if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                key=lambda k: k[0])
    starts = sorted(e["ts"] for e in ev if e.get("cat") == "kernel" and MLA in e["name"])
    steps, cur = [], [starts[0]]
    for t in starts[1:]:
        if t - cur[-1] > 1000:
            steps.append(cur[0])
            cur = []
        cur.append(t)
    steps.append(cur[0])
    totals = [[] for _ in range(len(CATS) + 1)]
    periods = []
    for a, b in zip(steps, steps[1:]):
        pts = sorted({a, b} | {t for s, e, _ in ks if e > a and s < b for t in (max(s, a), min(e, b))})
        acc = [0.0] * (len(CATS) + 1)
        live = [k for k in ks if k[1] > a and k[0] < b]
        for x, y in zip(pts, pts[1:]):
            mid = (x + y) / 2
            active = [c for s, e, c in live if s <= mid < e]
            acc[min(active) if active else len(CATS)] += y - x
        for i, v in enumerate(acc):
            totals[i].append(v)
        periods.append(b - a)
    p = st.median(periods)
    print(f"{len(periods)} steps, median period {p / 1e3:.2f} ms (mean of parts below)")
    names = [c for c, _ in CATS] + ["GPU idle"]
    for n, v in sorted(zip(names, (st.mean(t) for t in totals)), key=lambda x: -x[1]):
        print(f"  {v / 1e3:6.2f} ms  {100 * v / st.mean(periods):5.1f}%  {n}")


if __name__ == "__main__":
    main()
