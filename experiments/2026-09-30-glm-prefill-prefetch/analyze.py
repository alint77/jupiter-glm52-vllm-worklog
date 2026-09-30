#!/usr/bin/env python3
"""Prefetch-slot A/B: prefill time by new-token bin and decode step time, as
within-node differences against a reference arm, pooled over nodes.

    analyze.py [--ref s1] [--rows-dir DIR]
Rows are rows-pf-<arm>-n<node>.jsonl from agentic_bench.py.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BINS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 1 << 30)]


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if "prefill_s" in r and not r.get("profiled")]


def new_tokens(r: dict) -> float:
    if r.get("queried_tokens"):
        return r["queried_tokens"] - r.get("cached_tokens", 0)
    return float("nan")


def summarise(rows: list[dict]) -> dict:
    out = {}
    nt = np.array([new_tokens(r) for r in rows])
    pf = np.array([r["prefill_s"] * 1000 for r in rows])
    for lo, hi in BINS:
        m = (nt >= lo) & (nt < hi)
        out[f"pf{lo}"] = (pf[m].mean() if m.any() else np.nan, int(m.sum()))
    dec = [r for r in rows if r.get("steps", 0) >= 20 and r.get("completion_tokens", 0) < 8192]
    steps = sum(r["steps"] for r in dec)
    out["step_ms"] = (sum(r["decode_s"] for r in dec) * 1000 / steps if steps else np.nan, len(dec))
    out["tps"] = (np.mean([r["tokens_per_step"] for r in dec]) if dec else np.nan, len(dec))
    out["prefill_total_s"] = (pf.sum() / 1000, len(rows))
    out["new_tokens"] = (np.nansum(nt), len(rows))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", type=Path,
                    default=Path("/e/fscratch/profound/naeimitabiei1/agentic-bench"))
    ap.add_argument("--ref", default="s1")
    args = ap.parse_args()
    by_node: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(args.rows_dir.glob("rows-pf-*-n*.jsonl")):
        m = re.match(r"rows-pf-(\w+)-n(\d+)$", path.stem)
        if m:
            rows = load(path)
            if rows:
                by_node[m.group(2)][m.group(1)] = summarise(rows)
    keys = [f"pf{lo}" for lo, _ in BINS] + ["step_ms", "tps", "prefill_total_s"]
    print("per node (value, n):")
    for node, arms in sorted(by_node.items(), key=lambda x: int(x[0])):
        for arm, s in sorted(arms.items()):
            cells = " ".join(f"{k}={s[k][0]:.1f}/{s[k][1]}" for k in keys)
            print(f"  n{node} {arm:4s} {cells}")
    for ref in (args.ref, "s2"):
        print(f"\nwithin-node difference vs {ref} (mean over nodes, [min, max], nodes):")
        diffs: dict[tuple[str, str], list[float]] = defaultdict(list)
        for arms in by_node.values():
            if ref not in arms:
                continue
            for arm, s in arms.items():
                if arm == ref:
                    continue
                for k in keys:
                    a, b = s[k][0], arms[ref][k][0]
                    if np.isfinite(a) and np.isfinite(b):
                        diffs[(arm, k)].append(a - b)
        for (arm, k), d in sorted(diffs.items()):
            d = np.array(d)
            print(f"  {arm:4s} {k:16s} {d.mean():+8.2f}  [{d.min():+.2f}, {d.max():+.2f}]  {len(d)}")


if __name__ == "__main__":
    main()
