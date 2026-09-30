#!/usr/bin/env python3
"""Short prefetch A/B: median TTFT per prompt length and decode step time,
per node and arm, with each arm's within-node difference against s1."""

import json
import re
from collections import defaultdict
from pathlib import Path

OUT = Path("/e/fscratch/profound/naeimitabiei1/agentic-bench")
LENGTHS = (512, 1024, 2048, 4096)

table: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
for path in OUT.glob("prefill-pfs-*-n*-*.json"):
    m = re.match(r"prefill-pfs-(\w+)-n(\d+)-(\d+)$", path.stem)
    if m:
        table[(m.group(2), m.group(1))][m.group(3)] = json.loads(path.read_text())[
            "median_ttft_ms"]
for path in OUT.glob("rows-pfs-*-n*.jsonl"):
    m = re.match(r"rows-pfs-(\w+)-n(\d+)$", path.stem)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    dec = [r for r in rows if r.get("steps", 0) >= 10]
    if m and dec:
        steps = sum(r["steps"] for r in dec)
        table[(m.group(2), m.group(1))]["step"] = 1000 * sum(r["decode_s"] for r in dec) / steps
        table[(m.group(2), m.group(1))]["tps"] = sum(
            r["accepted"] for r in dec) / steps + 1
cols = [str(n) for n in LENGTHS] + ["step", "tps"]
print("node arm   " + " ".join(f"{c:>8s}" for c in cols) + "   (TTFT median ms; decode ms/step)")
for (node, arm), v in sorted(table.items()):
    base = table.get((node, "s1"), {})
    cells = []
    for c in cols:
        x = v.get(c)
        d = "" if arm == "s1" or x is None or c not in base or c == "tps" else f"({x - base[c]:+.1f})"
        cells.append(f"{x:8.1f}{d}" if x is not None else f"{'-':>8s}")
    print(f"n{node}   {arm:5s} " + " ".join(cells))
