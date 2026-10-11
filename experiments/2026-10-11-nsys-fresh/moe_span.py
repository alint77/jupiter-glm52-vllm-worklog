"""Per-layer wall span of the prefill MoE (route start -> combine end) and of
its stages, from an nsys report (one device), vs the summed kernel durations
(which double-count PDL-overlapped launches).

    moe_span.py report.sqlite [--device N]
"""
import argparse
import re
import sqlite3
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("--device", type=int, default=None)
a = ap.parse_args()
c = sqlite3.connect(a.report)
names = dict(c.execute("select id, value from StringIds"))
dev = a.device if a.device is not None else c.execute(
    "select min(deviceId) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
rows = c.execute("select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL "
                 "where deviceId=? order by start", (dev,)).fetchall()
layers, cur = [], None
for s, e, dn in rows:
    n = names[dn]
    if "tiered_prefill::" not in n:
        continue
    k = re.sub(r"<.*|\(.*", "", n.replace("void ", "")).split("::")[-1]
    if k == "route_kernel":
        cur = []
        layers.append(cur)
    if cur is not None:
        cur.append((s, e, k))
spans, sums, stage = [], [], {}
for L in layers:
    if not L or L[-1][2] != "combine_kernel":
        continue
    spans.append((L[-1][1] - L[0][0]) / 1e3)
    sums.append(sum(e - s for s, e, _ in L) / 1e3)
    t = {}
    for s, e, k in L:
        b, f = t.get(k, (s, e))
        t[k] = (min(b, s), max(f, e))
    for k, (b, f) in t.items():
        stage.setdefault(k, []).append((f - b) / 1e3)
print(f"layers {len(spans)}  span/layer median {st.median(spans):.0f} us  "
      f"total {sum(spans)/1e3:.1f} ms  (sum of kernel durations {sum(sums)/1e3:.1f} ms)")
for k, v in stage.items():
    print(f"  {k:18s} span median {st.median(v):7.1f} us")
