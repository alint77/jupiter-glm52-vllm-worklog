"""Trace analysis for the qualified MTP3-overlap no-SP config (job 969649).

Segments each decode step into target-verify and draft passes, maps kernels
to model components, and quantifies busy/idle/overlap per stream.
"""

import json
import re
import sys
import collections

TRACE_DIR = "traces"

COMPONENT_RULES = [
    # (regex, component, model part)
    (r"marlin_moe_wna16::Marlin", "routed_moe_marlin", "target routed experts (W4 Marlin, hot HBM + cold Grace)"),
    (r"cross_device_reduce", "custom_allreduce", "TP4 all-reduce (custom one-stage)"),
    (r"machete::Mac", "dense_w4_machete", "target dense W4 (Q-B / O / shared+dense MLP)"),
    (r"^void marlin::Marlin", "dense_w4_marlin", "target fused QKV-A (W4 Marlin)"),
    (r"flash_fwd_splitkv_mla_fp8_sparse", "sparse_mla", "sparse FP8 MLA attention"),
    (r"flash_fwd_mla_combine", "sparse_mla", "sparse FP8 MLA attention"),
    (r"get_mla_metadata", "sparse_mla_meta", "sparse MLA scheduling metadata"),
    (r"sm90_fp8_paged_mqa_logits", "dsa_scan", "DSA indexer full-context K scan"),
    (r"cooperative_topk", "dsa_topk", "DSA indexer top-2048 selection"),
    (r"indexer_k_quant|_fused_indexer_q_rope_quant|fp8_blockscale_gemm|cp_gather_indexer", "dsa_other", "DSA indexer projections/quant/cache"),
    (r"moe_sum_vec_kernel|act_and_mul_kernel|moe_align_block_size|count_and_sort_expert", "moe_support", "routed MoE support (sort/align/silu/sum)"),
    (r"grouped_topk_fused", "router", "MoE router top-8"),
    (r"ll_bf16_dotprodLLBf16Dotprod", "router_gemm", "MoE router GEMM (cute dsl)"),
    (r"nvjet_sm90_tst_192x8", "vocab_gemm", "vocabulary projection GEMM"),
    (r"nvjet_sm90", "mla_bf16_gemm", "MLA W_UK/W_UV BF16 contractions"),
    (r"ncclDevKernel_AllGather", "nccl_allgather", "vocabulary NCCL all-gather"),
    (r"ncclDevKernel", "nccl_other", "other NCCL"),
    (r"deep_gemm::fp8_gemm_kernel", "mtp_fp8_gemm", "MTP draft FP8 dense/MoE GEMMs"),
    (r"deep_gemm::sm90_fp8_gemm|fp8_gemm", "mtp_fp8_gemm", "MTP draft FP8 GEMMs"),
    (r"triton_tem_fused_mm", "mtp_bf16_proj", "MTP eh_proj/embed BF16 GEMMs"),
    (r"concat_and_cache_ds_mla|ConcatMLAQKernel|_convert_req_index", "kv_write", "KV-cache write/index"),
    (r"triton_|elementwise|FillFunctor|CUDAFunctor|memcpy32_post|CatArrayBatchedCopy|reduce_kernel|index_elementwise|masked_fill|scatter|gather|arange|cumsum|sort|Copy|copy_", "elementwise_meta", "elementwise / norm / metadata"),
]
COMPILED = [(re.compile(p), c, m) for p, c, m in COMPONENT_RULES]


def classify(name):
    for pat, comp, part in COMPILED:
        if pat.search(name):
            return comp, part
    return "other", "unclassified"


def union_busy(intervals):
    if not intervals:
        return 0.0, []
    ivs = sorted(intervals)
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged), merged


def gaps_between(merged, lo, hi, min_gap=50.0):
    """Idle gaps (us) inside [lo, hi] given merged busy intervals."""
    out = []
    prev = lo
    for s, e in merged:
        if s > prev and s - prev >= min_gap:
            out.append((prev, s))
        prev = max(prev, e)
    if hi > prev and hi - prev >= min_gap:
        out.append((prev, hi))
    return out


def analyze_rank(path, rank, report):
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")]
    for e in kernels:
        e["end"] = e["ts"] + e["dur"]

    # step boundaries from CPU-side generation annotations
    steps = sorted(
        (e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")),
        key=lambda e: e["ts"],
    )
    step_starts = [s["ts"] for s in steps]
    # skip the first step (warm-in); steady steps 1..7 plus a synthetic end
    bounds = step_starts + [max(e["end"] for e in kernels)]

    per_step = []
    comp_acc = collections.defaultdict(float)
    comp_cnt = collections.defaultdict(int)
    phase_acc = collections.defaultdict(list)
    stream_acc = collections.defaultdict(float)

    for si in range(1, len(bounds) - 1):
        lo, hi = bounds[si], bounds[si + 1]
        ks = [e for e in kernels if lo <= e["ts"] < hi]
        if not ks:
            continue
        wall_lo = min(e["ts"] for e in ks)
        wall_hi = max(e["end"] for e in ks)
        step_wall = hi - lo

        ivs = [(e["ts"], e["end"]) for e in ks]
        busy, merged = union_busy(ivs)
        idle_windows = gaps_between(merged, wall_lo, wall_hi)
        idle_in_span = sum(e - s for s, e in idle_windows)
        lead_gap = wall_lo - lo  # CPU time before first kernel of the step

        streams = collections.defaultdict(list)
        for e in ks:
            streams[e["args"]["stream"]].append((e["ts"], e["end"]))
            stream_acc[e["args"]["stream"]] += e["dur"]

        # phase segmentation: vocab GEMMs mark end of target and each draft pass
        vocab = sorted((e for e in ks if "nvjet_sm90_tst_192x8" in e["name"]), key=lambda e: e["ts"])
        phases = {}
        if len(vocab) == 4:
            cut0 = wall_lo
            names = ["target_verify", "draft1", "draft2", "draft3"]
            for pname, v in zip(names, vocab):
                phases[pname] = (cut0, v["end"])
                cut0 = v["end"]
            phases["tail"] = (cut0, wall_hi)
        for pname, (plo, phi) in phases.items():
            pks = [(e["ts"], e["end"]) for e in ks if plo <= e["ts"] < phi]
            pbusy, pmerged = union_busy(pks)
            pidle = (phi - plo) - pbusy
            phase_acc[pname].append((phi - plo, pbusy, pidle))

        for e in ks:
            comp, _ = classify(e["name"])
            comp_acc[comp] += e["dur"]
            comp_cnt[comp] += 1

        per_step.append(
            dict(step=si, wall=step_wall, kernel_span=wall_hi - wall_lo,
                 busy=busy, idle=idle_in_span, lead=lead_gap,
                 idle_windows=idle_windows, n_kernels=len(ks))
        )

    n = len(per_step)
    report.append(f"\n=== rank {rank}: {n} steady steps ===")
    w = sum(s["wall"] for s in per_step) / n
    span = sum(s["kernel_span"] for s in per_step) / n
    busy = sum(s["busy"] for s in per_step) / n
    idle = sum(s["idle"] for s in per_step) / n
    lead = sum(s["lead"] for s in per_step) / n
    report.append(
        f"step wall {w/1000:.3f} ms | kernel span {span/1000:.3f} | union busy {busy/1000:.3f} "
        f"| idle-in-span {idle/1000:.3f} | lead gap {lead/1000:.3f}"
    )

    report.append("\nphases (mean over steps): span | union busy | idle")
    for pname in ("target_verify", "draft1", "draft2", "draft3", "tail"):
        vals = phase_acc.get(pname)
        if not vals:
            continue
        ps = sum(v[0] for v in vals) / len(vals)
        pb = sum(v[1] for v in vals) / len(vals)
        pi = sum(v[2] for v in vals) / len(vals)
        report.append(f"  {pname:14s} {ps/1000:7.3f} ms {pb/1000:7.3f} ms {pi/1000:7.3f} ms")

    report.append("\nper-stream busy ms/step:")
    for s in sorted(stream_acc):
        report.append(f"  stream {s}: {stream_acc[s]/1000/n:7.3f}")

    report.append("\ncomponents ms/step (calls/step):")
    for comp, dur in sorted(comp_acc.items(), key=lambda kv: -kv[1]):
        report.append(f"  {comp:22s} {dur/1000/n:8.3f}  ({comp_cnt[comp]/n:.0f})")

    # biggest idle windows for attribution
    allw = []
    for s in per_step:
        for a, b in s["idle_windows"]:
            allw.append((b - a, a, b, s["step"]))
    allw.sort(reverse=True)
    report.append("\nlargest idle windows (ms, step):")
    for d, a, b, st in allw[:6]:
        report.append(f"  {d/1000:7.3f} ms in step {st} at +{(a - bounds[st])/1000:.3f} ms")
    return per_step, comp_acc, phase_acc, n


def main():
    report = []
    summary = {}
    for rank in range(4):
        per_step, comp, phases, n = analyze_rank(f"{TRACE_DIR}/rank{rank}.json", rank, report)
        summary[rank] = (per_step, comp, phases, n)
    print("\n".join(report))
    return summary


if __name__ == "__main__":
    main()
