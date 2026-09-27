#!/usr/bin/env python3
"""Speed-of-light table for MiMo-V2.6 decode (TP4/EP4, 8-token DFlash verify).

Every kernel family in the decode step is mapped to its role and to the bytes
it must move per call, from the checkpoint's real shapes and dtypes, per GPU:

  qkv_proj   fp8  27136x6144 / 4 ranks            41.7 MB
  o_proj     bf16 6144x16384 / 4 (not quantized)  50.3 MB
  router     bf16 384x6144, replicated              4.7 MB
  dense MLP  fp8  layer 0 gate_up 8192x6144, down 6144x4096 per rank
  lm_head    bf16 152576x6144 / 4                 468.8 MB
  attention  bf16 KV, 2 KV heads x (192 + 128) per rank; 60 layers read a
             128-token window, 10 read the whole context

At 8 tokens every GEMM here has arithmetic intensity ~8-16 FLOP/byte, far
below the H100 ridge (~270 for bf16 at 3.6 TB/s), so each is priced at its
byte floor: bytes / measured HBM stream bandwidth (3.63 TB/s, 2026-08-31
probe). Kernels too small to be bandwidth-bound are reported against the
latency floor instead (the cross-rank minimum for collectives).

Routed Marlin is excluded here; its bytes depend on how many experts each
rank runs, which `marlin_by_count.py` measures.

    sol.py <trace-dir> [--context TOKENS] [--json out.json]
"""

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

from analyze import RANK_RE, load, steps

HBM_BPS = 3.6257e12
MB = 1e6

QKV = 6784 * 6144 + (6784 // 128) * (6144 // 128) * 4
O_PROJ = 6144 * 4096 * 2
ROUTER = 384 * 6144 * 2
MLP_GATE_UP = 8192 * 6144 + (8192 // 128) * (6144 // 128) * 4
MLP_DOWN = 6144 * 4096 + (6144 // 128) * (4096 // 128) * 4
LM_HEAD = 38144 * 6144 * 2
KV_PER_TOKEN = 2 * (192 + 128) * 2  # 2 local KV heads, K 192 + V 128, bf16
SWA_WINDOW = 128 + 8


def roles(context: int):
    """(role, matcher, bytes per call or None when latency-bound, flops)."""
    return [
        ("MoE one-kernel w13 (both tiers)", lambda n, g: "tiered_decode" in n and "gemm_kernel<0>" in n, None),
        ("MoE one-kernel w2 (both tiers)", lambda n, g: "tiered_decode" in n and "gemm_kernel<1>" in n, None),
        ("MoE one-kernel route / act / finalize", lambda n, g: "tiered_decode" in n, None),
        ("MoE Marlin hot", lambda n, g: "marlin_moe" in n and g == 264, None),
        ("MoE Marlin cold", lambda n, g: "marlin_moe" in n and g == 132, None),
        ("qkv_proj (fp8)", lambda n, g: "fp8_gemm_kernel_swapAB<6784u, 6144u" in n, QKV),
        ("dense MLP gate_up (fp8)", lambda n, g: "fp8_gemm_kernel_swapAB<8192u, 6144u" in n, MLP_GATE_UP),
        ("dense MLP down (fp8)", lambda n, g: "fp8_gemm_kernel_swapAB<6144u, 4096u" in n, MLP_DOWN),
        ("o_proj (bf16)", lambda n, g: n.startswith("nvjet_sm90_tst_64x8_64x16_4x1") and g == 96, O_PROJ),
        ("router gate (bf16)", lambda n, g: "nvjet_sm90_tss_64x8_64x16_4x1_v_bz_splitK" in n, ROUTER),
        ("lm_head (bf16)", lambda n, g: "nvjet_sm90_tst_192x8" in n, LM_HEAD),
        ("attention, full (FA3)", lambda n, g: "FlashAttnFwdSm90" in n and "Combine" not in n,
         context * KV_PER_TOKEN),
        ("attention, sliding (FA4)", lambda n, g: "flash_fwd_sm90FlashAttentionForwardSm90" in n,
         SWA_WINDOW * KV_PER_TOKEN),
        ("replica assign + align", lambda n, g: "_assign_kernel" in n, None),
        ("TP all-reduce", lambda n, g: "cross_device_reduce" in n, None),
        ("MoE sum / act / topk", lambda n, g: any(k in n for k in (
            "moe_sum", "act_and_mul", "grouped_topk")), None),
        ("norms / rope / elementwise", lambda n, g: n.startswith("triton_") or "elementwise" in n
         or "rms_norm" in n or "rotary" in n, None),        ("DFlash drafter (all but lm_head)", lambda n, g: False, None),
    ]


def analyze(trace_dir: Path, context: int) -> dict:
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    n_steps = {}
    for path in sorted(trace_dir.glob("*.pt.trace.json.gz")):
        rank = int(RANK_RE.search(path.name).group(1))
        rows = steps(load(path))
        n_steps[rank] = len(rows)
        for row in rows:
            for phase, ops in row["phases"].items():
                for op in ops:
                    grid = op["args"].get("grid") or [0]
                    grid = grid[0] * (grid[1] if len(grid) > 1 else 1)
                    matched = None
                    for role, match, _ in roles(context):
                        if match(op["name"], grid):
                            matched = role
                            break
                    # The drafter's own GEMMs and attention (5 layers, 1024-token
                    # window) are reported as one row; lm_head keeps its role.
                    if phase == "draft" and matched != "lm_head (bf16)":
                        matched = "DFlash drafter (all but lm_head)"
                    if matched:
                        table[matched][rank].append(op["end"] - op["t"])
    out = {}
    for role, _, nbytes in roles(context):
        per_rank = table.get(role, {})
        if not per_rank:
            continue
        calls = statistics.fmean(len(v) / n_steps[r] for r, v in per_rank.items())
        mean_us = statistics.fmean(statistics.fmean(v) for v in per_rank.values()) * 1000
        ms_step = statistics.fmean(sum(v) / n_steps[r] for r, v in per_rank.items())
        row = {"calls_per_step": calls, "mean_us": mean_us, "ms_per_step": ms_step}
        if nbytes:
            sol_us = nbytes / HBM_BPS * 1e6
            row |= {"mb_per_call": nbytes / MB, "sol_us": sol_us,
                    "achieved_tbps": nbytes / (mean_us * 1e-6) / 1e12,
                    "efficiency": sol_us / mean_us,
                    "sol_ms_per_step": sol_us * calls / 1000}
        out[role] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--context", type=int, nargs="+", required=True,
                        help="approximate KV length per trace dir, in the same order")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = {}
    for trace_dir, context in zip(args.dirs, args.context):
        result = analyze(trace_dir, context)
        results[trace_dir.name] = result
        print(f"\n===== {trace_dir.name} (context ~{context} tokens)")
        print(f"{'role':28s} {'calls':>6s} {'mean us':>8s} {'ms/step':>8s} "
              f"{'MB/call':>8s} {'SOL us':>7s} {'TB/s':>6s} {'% SOL':>6s}")
        for role, row in result.items():
            extra = ""
            if "sol_us" in row:
                extra = (f"{row['mb_per_call']:8.1f} {row['sol_us']:7.1f} "
                         f"{row['achieved_tbps']:6.2f} {row['efficiency'] * 100:5.0f}%")
            print(f"{role:28s} {row['calls_per_step']:6.1f} {row['mean_us']:8.1f} "
                  f"{row['ms_per_step']:8.3f} {extra}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
