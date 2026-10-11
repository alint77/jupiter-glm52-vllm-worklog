"""Skip-KV staging cost from an Nsight Systems report (graph kernels per node).

Per verify step on one device:
- stall: a skip layer's FlashMLA starting late because the staged rows were not
  ready (gap after the main stream's previous kernel while the group's gather
  was still running);
- MoE contention: within each layer position, the slope of layer_kernel
  duration against its overlap with staging kernels (per-position means
  removed, so layer identity does not confound it).

    staging_cost.py report.sqlite [--device N]
"""
import argparse
import collections
import sqlite3
import statistics as st

STAGE = ("_gather_rows_kernel", "_mark_rows_kernel", "_compact_rows_kernel", "_remap_rows_kernel")

ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("--device", type=int, default=0)
a = ap.parse_args()
c = sqlite3.connect(a.report)
names = dict(c.execute("select id, value from StringIds"))
by_corr = collections.defaultdict(list)
for s, e, corr, n, stream in c.execute(
        "select start, end, correlationId, demangledName, streamId from "
        "CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ?", (a.device,)):
    by_corr[corr].append((s, e, names[n], stream))
steps = [sorted(ks) for ks in by_corr.values() if len(ks) >= 500]
steps.sort()
stalls, gathers, n_stall = [], [], 0
pos_rows = collections.defaultdict(list)
streams = collections.Counter()
for ks in steps:
    g = [k for k in ks if "_gather_rows_kernel" in k[2]]
    stage = [k for k in ks if any(p in k[2] for p in STAGE)]
    fa = [k for k in ks if "flash_fwd_splitkv_mla_fp8_sparse" in k[2]]
    # graph replay spreads nodes over arbitrary streams: "busy" is any
    # non-staging kernel
    mk = [k for k in ks if not any(p in k[2] for p in STAGE)]
    total = 0.0
    for gs, ge, _, _ in g:
        gathers.append((ge - gs) / 1e3)
        nxt = next((f for f in fa if f[0] > gs), None)
        if nxt is None:
            continue
        prev = max((k[1] for k in mk if k[0] < nxt[0] and k is not nxt), default=None)
        if prev is not None and ge > prev and nxt[0] - ge < 5e3:
            total += (ge - prev) / 1e3
            n_stall += 1
    stalls.append(total)
    lk = [k for k in ks if "layer_kernel" in k[2]]
    if len(lk) != 75:
        continue
    for i, (s, e, _, _) in enumerate(lk):
        ov = sum(max(0, min(e, se) - max(s, ss)) for ss, se, _, _ in stage) / 1e3
        pos_rows[i].append((len(stalls) - 1, (e - s) / 1e3, ov))
# two-way demeaning (layer position and step): routing makes some steps
# heavier for every layer, and those may also stage more rows
obs = [r + (i,) for i, rows in pos_rows.items() for r in rows]
y = {(s, i): d for s, d, _, i in obs}
x = {(s, i): o for s, _, o, i in obs}
for v in (y, x):
    for _ in range(10):
        for key, grp in ((0, collections.defaultdict(list)), (1, collections.defaultdict(list))):
            for k, val in v.items():
                grp[k[key]].append(val)
            m = {g: st.mean(vals) for g, vals in grp.items()}
            for k in v:
                v[k] -= m[k[key]]
slope = sum(x[k] * y[k] for k in x) / max(sum(x[k] ** 2 for k in x), 1e-9)
ov_total = sum(o for _, _, o, _ in obs) / max(len({s for s, _, _, _ in obs}), 1)
print(f"steps {len(steps)}  gather mean {st.mean(gathers):.1f} us (p90 "
      f"{sorted(gathers)[int(0.9 * len(gathers))]:.1f})  stall {st.mean(stalls) / 1e3:.3f} ms/step "
      f"({n_stall / len(steps):.1f} late groups/step)  layer_kernel overlap with staging "
      f"{ov_total / 1e3:.2f} ms/step, slope {slope:+.3f} -> contention ~{slope * ov_total / 1e3:+.3f} ms/step")
