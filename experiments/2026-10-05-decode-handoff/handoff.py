#!/usr/bin/env python3
"""Handoff summary of gap_probe .pt traces: per cell, medians (us) relative
to w13's end of act end, the w2 producers' PDL return (rows can load), and
the w2 consumers' first full stage; plus the call length.
Usage: handoff.py <dir> [<dir> ...]"""
import glob
import statistics as st
import sys
from pathlib import Path

import torch

src = (Path(__file__).resolve().parents[1] / "2026-10-04-c2c-roofline/gap_probe.py").read_text()
ns: dict = {}
exec(src[src.index("def split_calls"):src.index("def timeline")], ns)

for d in sys.argv[1:]:
    for path in sorted(glob.glob(f"{d}/*.pt")):
        rows = []
        for s0, s1, rs in ns["split_calls"](torch.load(path))[5:-5]:
            w13 = max(r["t2"] for r in rs if r["ph"] in (0, 10))
            act = max(r["t2"] for r in rs if r["ph"] in (53, 63))
            pdl = min(r["t1"] for r in rs if r["ph"] == 61 and r["t1"])
            c2 = min(r["t0"] for r in rs if r["ph"] == 71)
            rows.append(((act - w13) / 1e3, (pdl - w13) / 1e3, (c2 - w13) / 1e3, (s1 - s0) / 1e3))
        m = [round(st.median(r[i] for r in rows), 2) for i in range(4)]
        print(f"{Path(d).name:>14}/{Path(path).stem:6s} act_end {m[0]:5}  w2_pdl {m[1]:5}  "
              f"w2_first_stage {m[2]:5}  call {m[3]:6}")
