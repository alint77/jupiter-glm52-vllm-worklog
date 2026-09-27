#!/usr/bin/env python3
"""Tiered kernel vs production Marlin at the pinned base clock, per (hot, cold) cell.

    compare_fixedclock.py <sweep dir> [--json out.json]

Marlin per layer = max(hot tier, cold tier), each tier = w13 + act_and_mul + w2
+ moe_sum kernels (align excluded: production fuses it into replica routing).
max() assumes perfect two-stream overlap, which production nearly reaches, so
this is the best case for Marlin. Tiered = w13 + w2 kernels. Cells are weighted
by their probability in the held-out trace (count-distribution.txt).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ncu_sum import marlin, sk

HERE = Path(__file__).resolve().parent


def cells():
    out = {}
    for ln in (HERE / "count-distribution.txt").read_text().splitlines():
        m = re.match(r"\s*hot\s+(\d+) cold (\d+):\s+([\d.]+)%", ln)
        if m:
            out[(int(m[1]), int(m[2]))] = float(m[3]) / 100
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    d = args.dir
    m_hot = {int(p.stem.split("-")[2]): marlin(p) for p in d.glob("marlin-hot-*.csv") if p.stat().st_size}
    m_cold = {int(p.stem.split("-")[2]): marlin(p) for p in d.glob("marlin-cold-*.csv") if p.stat().st_size}
    tier = {}
    for p in d.glob("sk-*.csv"):
        if not p.stat().st_size:
            continue
        _, g, h, c, _cc = p.stem.split("-")
        tier[(g, int(h), int(c))] = sk(p)["us"]

    def interp(table, n):
        if n == 0:
            return 0.0
        ks = sorted(table)
        return float(np.interp(n, ks, [table[k]["total"] for k in ks]))

    def tiered(h, c):
        tot = 0.0
        for g in ("w13", "w2"):
            hs = sorted(k[1] for k in tier if k[0] == g and k[2] == c)
            if not hs:
                return None
            tot += float(np.interp(h, hs, [tier[(g, x, c)] for x in hs]))
        return tot

    print(f"{'cell':>12s} {'p':>6s} {'Marlin':>8s} {'tiered':>8s} {'speedup':>8s}")
    rows, wm, wt, wp = [], 0.0, 0.0, 0.0
    for (h, c), p in sorted(cells().items(), key=lambda kv: -kv[1]):
        mt = max(interp(m_hot, h), interp(m_cold, c))
        tt = tiered(h, c)
        if tt is None:
            continue
        rows.append({"hot": h, "cold": c, "p": p, "marlin_us": mt, "tiered_us": tt})
        wm += p * mt; wt += p * tt; wp += p
        print(f"{f'({h:2d},{c})':>12s} {p:6.2%} {mt:8.1f} {tt:8.1f} {mt / tt:8.2f}x")
    print(f"weighted over {wp:.0%} of cases: Marlin {wm / wp:.1f} us, tiered {wt / wp:.1f} us, "
          f"{wm / wt:.2f}x; per step (69 layers, one pass) saves {(wm - wt) / wp * 69 / 1000:.2f} ms")
    for n, v in sorted(m_hot.items()):
        print(f"  marlin hot {n:2d}: w13 {v['w13']:.1f} act {v['act']:.1f} w2 {v['w2']:.1f} sum {v['sum']:.1f}"
              f" total {v['total']:.1f} (align {v['align']:.1f})")
    for n, v in sorted(m_cold.items()):
        print(f"  marlin cold {n}: w13 {v['w13']:.1f} act {v['act']:.1f} w2 {v['w2']:.1f} sum {v['sum']:.1f}"
              f" total {v['total']:.1f}")
    if args.json:
        args.json.write_text(json.dumps({"cells": rows, "marlin_hot": m_hot, "marlin_cold": m_cold,
                                         "tiered": {f"{g}-{h}-{c}": v for (g, h, c), v in tier.items()}},
                                        indent=1))


if __name__ == "__main__":
    main()
