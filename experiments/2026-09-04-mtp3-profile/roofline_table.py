#!/usr/bin/env python3
"""Roofline for each prefill role, from shapes joined in prefill_roofline.py.

Ceilings are GH200 120GB *spec* values. The project's earlier roofline used
conservative ones (HBM 3.5 TB/s, BF16 630 TF/s, FP8 1260 TF/s); against those,
five rows here exceed 100%, which is a statement about the ceiling rather than
the kernel, so spec is used and the conservative equivalent is noted.

A row's applicable roof is min(compute ceiling, AI x bandwidth). FLOPs and
logical bytes are derived from observed shapes plus the config; they are not
measured DRAM transactions. Rows flagged (b) have a byte model that is a bound
rather than an estimate, so their AI and roof are indicative only -- the
achieved FLOP/s and the millisecond columns are unaffected.
"""
import json
from pathlib import Path

HBM, C2C_SPEC, C2C_MEAS = 4.0e12, 450e9, 373e9
BF16, FP8 = 989e12, 1979e12
NVL_AGG = 450e9
L, M, TOPK, EP = 75, 8192, 8, 4
ROWS = M * TOPK / EP
E_HOT, E_COLD = 38.9, 25.1

blob = json.loads(Path("agent_space/experiments/2026-09-04-mtp3-profile/prefill-roles.json").read_text())
R = {r["op"] + "|" + r["tag"]: r for r in blob["roles"]}
BUSY = blob["busy"]
out = []

def add(name, ms, f, b, bw, peak, flag=""):
    out.append(dict(name=name, ms=ms, f=f, b=b, bw=bw, peak=peak, flag=flag))

for tier, E in (("hot", E_HOT), ("cold", E_COLD)):
    for gemm, (K, N, kp, npk, g, n2) in (("w13", (6144, 4096, 384, 8192, 192, 4096)),
                                         ("w2", (2048, 6144, 128, 12288, 64, 6144))):
        r = R[f"_moe_C::moe_wna16_marlin_gemm|{tier} {gemm}"]
        rows = ROWS * E / 64
        f = 2 * rows * K * N * L
        b = (E * kp * npk * 4 + E * g * n2 * 2) * L + (rows * K + rows * N) * 2 * L
        add(f"routed MoE Marlin W4 {tier} {gemm}", r["ms"], f, b,
            C2C_MEAS if tier == "cold" else HBM, BF16)

r = R["_flashmla_C::sparse_decode_fwd|"]; C = r["calls"]
f_full = 2 * M * 64 * 2048 * (576 + 512) * C
b_mla = M * 2048 * 656 * C
add("sparse MLA FP8, 64 heads launched", r["ms"], f_full, b_mla, HBM, FP8, "b")
add("  the 16 real TP heads only", r["ms"], f_full * 16 / 64, b_mla, HBM, FP8, "u")

r = R["vllm::sparse_attn_indexer|"]; C = r["calls"]; CTX = 3.5 * 8192
f = 2 * M * 32 * CTX * 128 * C
b = (CTX * 132 + M * 32 * 128 + M * CTX * 4) * C
add("DSA indexer FP8 scan", r["ms"], f, b, HBM, FP8, "b")

for tag, K, N in (("4096x6144", 4096, 6144), ("6144x2624", 6144, 2624),
                  ("2048x4096", 2048, 4096), ("6144x1024", 6144, 1024),
                  ("512x6144", 512, 6144)):
    r = R.get(f"aten::mm|{tag}")
    if r:
        C = r["calls"]
        add(f"BF16 dense GEMM {tag}", r["ms"], 2 * M * K * N * C,
            2 * (M * K + K * N + M * N) * C, HBM, BF16)

r = R["aten::bmm|"]; C = r["calls"]
add("BF16 MLA W_UK/W_UV bmm", r["ms"], 2 * 16 * M * 192 * 512 * C,
    2 * (16 * M * 192 + 16 * 192 * 512 + 16 * M * 512) * C, HBM, BF16)
r = R["_C::silu_and_mul|"]; C = r["calls"]
add("MoE activation silu_and_mul", r["ms"], None, (65536 * 4096 + 65536 * 2048) * 2 * C, HBM, BF16)
r = R["_C::top_k_per_row_prefill|"]; C = r["calls"]
add("DSA top-k per row", r["ms"], None, (4096 * 32768 * 4 + 4096 * 2048 * 4) * C, HBM, BF16)
r = R["vllm::all_reduce|"]; C = r["calls"]
add("TP all-reduce (NCCL ring, bus bytes)", r["ms"], None, 1.5 * M * 6144 * 2 * C, NVL_AGG, BF16)

print(f"{'role':37s} {'ms':>7s} {'%':>5s} {'AI':>9s} {'achieved':>10s} {'effBW':>8s} "
      f"{'roof':>9s} {'eff%':>6s}")
print("-" * 100)
FULL_MLA_ROOF = None
for r in out:
    t = r["ms"] / 1000
    ai = (r["f"] / r["b"]) if (r["f"] and r["b"]) else None
    ach = (r["f"] / t) if r["f"] else None
    ebw = r["b"] / t
    if r["f"]:
        roof = min(r["peak"], ai * r["bw"])
        if r["flag"] == "u":
            roof = FULL_MLA_ROOF
        else:
            FULL_MLA_ROOF = roof if "sparse MLA" in r["name"] else FULL_MLA_ROOF
        eff = 100 * ach / roof
        rs = f"{roof/1e12:8.0f}T"
    else:
        roof, eff, rs = r["bw"], 100 * ebw / r["bw"], f"{r['bw']/1e9:8.0f}G"
    print(f"{r['name'][:37]:37s} {r['ms']:7.1f} {100*r['ms']/BUSY:5.1f} "
          f"{(f'{ai:9.0f}' if ai else '        --')} "
          f"{(f'{ach/1e12:7.0f} TF' if ach else '        --')} "
          f"{ebw/1e9:7.0f}G {rs} {eff:6.1f}{r['flag'] and ' '+r['flag']}")
print("-" * 100)
print(f"spec ceilings: HBM {HBM/1e12:.1f} TB/s | C2C {C2C_MEAS/1e9:.0f} GB/s measured "
      f"({C2C_SPEC/1e9:.0f} spec) | NVLink 1-way {NVL_AGG/1e9:.0f} GB/s | "
      f"BF16 {BF16/1e12:.0f} TF/s | FP8 {FP8/1e12:.0f} TF/s")
print("(b) byte model is a bound, so AI/roof indicative; ms and achieved FLOP/s unaffected")
print("(u) useful-work row: real-head FLOPs against the *full* kernel's roof")
