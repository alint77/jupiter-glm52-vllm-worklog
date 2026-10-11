"""Step breakdown per concurrency (ms/step, mean over the 4 ranks) from
an/bd-c<c>-w<window>-d<dev>.json; window 1 = ~5K context, 2 = ~50K."""
import json
import os
import sys
from pathlib import Path

A = Path(os.environ.get("SOL_AN", Path(__file__).parent / "an"))
cs = [1, 2, 4, 8, 16]
w = sys.argv[1] if len(sys.argv) > 1 else "1"
d = {c: [json.loads((A / f"bd-c{c}-w{w}-d{dev}.json").read_text()) for dev in range(4)]
     for c in cs}
cats = list(d[1][0]["ms"])

mean = lambda v: sum(v) / len(v)
print("| category | " + " | ".join(f"c={c}" for c in cs) + " |")
print("| --- |" + " ---: |" * len(cs))
print("| **step period** | " + " | ".join(f"**{mean([r['period_ms'] for r in d[c]]):.2f}**" for c in cs) + " |")
for k in sorted(cats, key=lambda k: -mean([r["ms"].get(k, r.get("idle_ms", 0)) for r in d[16]])):
    vals = []
    for c in cs:
        vals.append(mean([r["ms"][k] if k in r["ms"] else r.get("idle_ms", 0.0) for r in d[c]]))
    print(f"| {k} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
