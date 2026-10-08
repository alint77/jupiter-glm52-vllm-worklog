"""Decode step breakdown for any drafter (one rank's torch-profiler trace).

Steps are the scheduler's execute_* annotations (the segmentation of
../2026-09-26-mimo-decode-profile/analyze.py, which holds for MTP's draft
passes as well as DFlash2's). Kernels launched by the step's largest CUDA
graph (the verify forward) get a category by name; everything else the step
launched (draft passes, lm_head, sampling, input prep) is "outside the verify
graph". Every instant from the step's first kernel to the next step's first
goes to the highest-priority category running then; none running is idle.

    breakdown2.py <trace.json.gz> [--json out.json]
"""
import argparse
import bisect
import collections
import gzip
import json
import statistics as st

MLA = "flash_fwd_splitkv_mla_fp8_sparse"
CATS = [
    ("MoE expert GEMMs", ["tiered_decode::gemm_kernel"]),
    ("MoE route/act/finalize + router", ["tiered_decode", "grouped_topk", "marlin_moe"]),
    ("attention (FlashMLA)", [MLA]),
    ("attention (split-KV combine)", ["flash_fwd_mla_combine"]),
    ("all-reduce + RMSNorm (incl. wait)", ["allreduce_fusion", "cross_device_reduce"]),
    ("DCP collectives", ["one_shot::"]),
    ("DSA indexer", ["mqa_logits", "cooperative_topk", "StableTopK", "indexer",
                     "_pack_dcp_topk", "convert_req_index"]),
    ("dense GEMMs (decode_gemm)", ["decode_gemm_kernel"]),
    ("dense GEMMs (cuBLAS etc.)", ["nvjet", "splitKreduce", "cutlass", "deep_gemm",
                                   "dotprod", "splitk", "gemm"]),
    ("KV write / skip-KV staging", ["concat_and_cache", "reshape_and_cache", "_gather_rows",
                                    "_mark_rows", "_compact_rows", "_remap_rows"]),
    ("other verify-graph kernels", [""]),
]
OUTSIDE = "outside the verify graph (draft, lm_head, sampling, prep)"
NAMES = [c for c, _ in CATS] + [OUTSIDE, "GPU idle"]
GPU = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH = {"cuda_runtime", "cuda_driver"}


def cat(name):
    for i, (_, keys) in enumerate(CATS):
        if any(k in name for k in keys):
            return i
    return len(CATS) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json")
    a = ap.parse_args()
    ev = json.load(gzip.open(a.trace))["traceEvents"]
    by_corr = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in GPU:
            by_corr[e.get("args", {}).get("correlation")].append(e)
    starts = sorted(e["ts"] for e in ev if e.get("cat") == "user_annotation"
                    and e["name"].startswith("execute_"))
    per_step = collections.defaultdict(list)
    for la in sorted((e for e in ev if e.get("cat") in LAUNCH), key=lambda e: e["ts"]):
        i = bisect.bisect(starts, la["ts"]) - 1
        if 0 <= i < len(starts) - 1:
            per_step[i].append(la)
    steps = []
    for i in sorted(per_step):
        graphs = [la for la in per_step[i] if "GraphLaunch" in la["name"]]
        if not graphs:
            continue
        target = max(graphs, key=lambda la: len(by_corr[la["args"]["correlation"]]))
        tcorr = target["args"]["correlation"]
        ks = []
        for la in per_step[i]:
            c = la["args"].get("correlation")
            for k in by_corr.get(c, []):
                ks.append((k["ts"], k["ts"] + k["dur"],
                           cat(k["name"]) if c == tcorr else len(CATS)))
        if len(by_corr[tcorr]) < 500:  # not a decode verify step
            continue
        steps.append(sorted(ks))
    rows = []
    for ks, nxt in zip(steps, steps[1:]):
        a0, b0 = ks[0][0], nxt[0][0]
        pts = sorted({a0, b0} | {t for s, e, _ in ks if e > a0 and s < b0
                                 for t in (max(s, a0), min(e, b0))})
        acc = [0.0] * len(NAMES)
        for x, y in zip(pts, pts[1:]):
            m = (x + y) / 2
            act = [c for s, e, c in ks if s <= m < e]
            acc[min(act) if act else len(NAMES) - 1] += y - x
        rows.append(acc)
    if not rows:
        print("no decode steps found")
        return
    mean = [st.mean(r[i] for r in rows) / 1e3 for i in range(len(NAMES))]
    total = sum(mean)
    print(f"{len(rows)} steps, mean period {total:.2f} ms")
    for n, v in sorted(zip(NAMES, mean), key=lambda x: -x[1]):
        print(f"  {v:6.2f} ms  {100 * v / total:5.1f}%  {n}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"steps": len(rows), "period_ms": total, "ms": dict(zip(NAMES, mean))}, f)


if __name__ == "__main__":
    main()
