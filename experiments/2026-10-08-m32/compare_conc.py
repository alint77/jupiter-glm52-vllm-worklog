"""c=4 served A/B from arm_conc.sh outputs: m8 = before (2af85c93c3), m32 =
after. Per case: step time per request (acceptance length / decode tok/s,
which cancels acceptance noise), per-request and aggregate tok/s; after -
before paired within each node, then averaged over nodes.

    compare_conc.py [root]
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/e/fscratch/profound/naeimitabiei1/m32")
logs = Path(__file__).resolve().parent
cell = collections.defaultdict(lambda: collections.defaultdict(list))  # (case, kind, node)
for d in sorted(root.glob("m*-*")):
    kind, job = d.name.split("-")[:2]
    log = logs / f"run-{d.name}.log"
    if not (d / "probe.jsonl").exists() or "=== done" not in log.read_text():
        continue
    for line in (d / "probe.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = (r["case"], r["n"])
        cell[key][(kind, job)].append(r)

print(f"{'case':18s} {'toks/step':>9s} | {'ms/step before':>14s} {'after':>7s} {'diff':>7s} | "
      f"{'agg tok/s before':>16s} {'after':>7s} | nodes")
for (case, n), arms in sorted(cell.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    nodes = sorted({job for _, job in arms})
    diffs, ms_b, ms_a, ag_b, ag_a = [], [], [], [], []
    for job in nodes:
        b, a = arms.get(("m8", job)), arms.get(("m32", job))
        if not b or not a:
            continue
        mb = st.mean(1000 * r["acc_len"] / r["decode_tps"] for r in b)
        ma = st.mean(1000 * r["acc_len"] / r["decode_tps"] for r in a)
        ms_b.append(mb)
        ms_a.append(ma)
        diffs.append(ma - mb)
        ag_b.append(st.mean(r["agg_tps"] for r in b))
        ag_a.append(st.mean(r["agg_tps"] for r in a))
    if not diffs:
        continue
    sd = st.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    print(f"{case:18s} {8 * n:9d} | {st.mean(ms_b):14.2f} {st.mean(ms_a):7.2f} "
          f"{st.mean(diffs):+7.2f} +- {sd:4.2f} | {st.mean(ag_b):16.1f} {st.mean(ag_a):7.1f} | "
          f"{len(diffs)}")
