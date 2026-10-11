"""Same-node paired step-time deltas (new - base) from the dcp-*.log files."""
import re
from collections import defaultdict
from pathlib import Path

step = defaultdict(list)
for f in Path(__file__).parent.glob("logs/dcp-*.log"):
    _, arm, node = f.stem.split("-")
    for m in re.finditer(r"^\s*(\d+) n=\s*(\d+) .* step\s+([\d.]+) ms", f.read_text(), re.M):
        step[(node, arm, int(m[1]), int(m[2]))].append(float(m[3]))
nodes = sorted({k[0] for k in step})
cells = sorted({k[2:] for k in step}, key=lambda c: (c[1], c[0]))
mean = lambda v: sum(v) / len(v)
print("| context | requests | base ms | new ms | " + " | ".join(f"node {n}" for n in nodes) + " | mean |")
print("| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in nodes) + " | ---: |")
for ctx, n in cells:
    d = [mean(step[(nd, "new", ctx, n)]) - mean(step[(nd, "base", ctx, n)]) for nd in nodes]
    b = mean([x for nd in nodes for x in step[(nd, "base", ctx, n)]])
    w = mean([x for nd in nodes for x in step[(nd, "new", ctx, n)]])
    print(f"| {ctx // 1000}K | {n} | {b:.2f} | {w:.2f} | " + " | ".join(f"{x:+.2f}" for x in d) + f" | **{mean(d):+.2f}** |")
