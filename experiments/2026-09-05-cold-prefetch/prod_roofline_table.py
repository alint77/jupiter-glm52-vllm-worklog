# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Roofline for the production checkpoint, GLM-5.2 AutoRound W4 group 64.

Three constants change against the 2026-09-04 table, and getting any of them
wrong silently rescales a column:

* **Scale groups.** That table hardcodes 192 and 64 scale groups per expert,
  which are `6144 // 32` and `2048 // 32` -- group size 32. Group 64 halves
  both. Scales are the smaller operand but the cold tier's whole point is that
  it reads them per tile.
* **Residency.** Per-expert normalisation uses hot/cold experts per layer.
  Production plans a different split, and budgeting the staging slot moves it
  again, so both come from the run's own log rather than a constant.
* **The cold roof.** The old table priced the cold tier against C2C, which was
  right when it read Grace. Staged, it reads HBM, and pricing it against C2C
  is what produced the nonsensical 89-94% efficiencies in PHASE3's table.

Model geometry is identical between the two checkpoints (78 layers, 75 routed,
hidden 6144, intermediate 2048, 256 experts, top-8), so everything else carries
over unchanged.
"""

import argparse
import json
from pathlib import Path

HBM, C2C_MEAS = 4.0e12, 373e9
BF16, FP8 = 989e12, 1979e12
NVL_AGG = 450e9
L, M, TOPK, EP = 75, 8192, 8, 4
ROWS = M * TOPK / EP
GROUP = 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", required=True, help="prefill_roofline.py --out JSON")
    ap.add_argument("--hot-experts", type=int, required=True, help="hot per rank")
    ap.add_argument("--cold-experts", type=int, required=True, help="cold per rank")
    ap.add_argument(
        "--cold-tier",
        choices=("hbm", "c2c"),
        default="hbm",
        help="Where the cold tier reads from: hbm when staged, c2c when not.",
    )
    args = ap.parse_args()

    blob = json.loads(Path(args.roles).read_text())
    R = {r["op"] + "|" + r["tag"]: r for r in blob["roles"]}
    e_hot, e_cold = args.hot_experts / L, args.cold_experts / L
    cold_bw = HBM if args.cold_tier == "hbm" else C2C_MEAS
    out = []

    for tier, E in (("hot", e_hot), ("cold", e_cold)):
        for gemm, (K, N, kp, npk, n2) in (
            ("w13", (6144, 4096, 384, 8192, 4096)),
            ("w2", (2048, 6144, 128, 12288, 6144)),
        ):
            key = f"_moe_C::moe_wna16_marlin_gemm|{tier} {gemm}"
            if key not in R:
                continue
            r = R[key]
            g = K // GROUP
            rows = ROWS * E / 64
            flops = 2 * rows * K * N * L
            byts = (E * kp * npk * 4 + E * g * n2 * 2) * L + (
                rows * K + rows * N
            ) * 2 * L
            out.append(
                (
                    f"routed MoE Marlin W4 {tier} {gemm}",
                    r["ms"],
                    flops,
                    byts,
                    HBM if tier == "hot" else cold_bw,
                )
            )

    busy = blob["busy"]
    print(
        f"{'role':<38}{'ms':>8}{'%':>7}{'AI':>9}{'achieved':>11}"
        f"{'effBW':>9}{'roof':>10}{'eff%':>7}"
    )
    print("-" * 99)
    for name, ms, f, b, bw in out:
        ai = f / b
        achieved = f / (ms * 1e-3)
        eff_bw = b / (ms * 1e-3)
        roof = min(BF16, ai * bw)
        print(
            f"{name:<38}{ms:8.1f}{100 * ms / busy:7.1f}{ai:9.0f}"
            f"{achieved / 1e12:8.0f} TF{eff_bw / 1e9:8.0f}G"
            f"{roof / 1e12:9.0f}T{100 * achieved / roof:7.1f}"
        )
    hot = sum(o[1] for o in out if " hot " in o[0])
    cold = sum(o[1] for o in out if " cold " in o[0])
    if hot and cold:
        print(
            f"\nper-expert: hot {1000 * hot / args.hot_experts:.1f} us, "
            f"cold {1000 * cold / args.cold_experts:.1f} us "
            f"({100 * (cold / args.cold_experts) / (hot / args.hot_experts) - 100:+.1f}% "
            "vs hot)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
