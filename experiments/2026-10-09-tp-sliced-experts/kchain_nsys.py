"""Per-kernel time per decode step and the median per-layer kernel chain of the
verify graph, from an Nsight Systems report (graph kernels per node).

    kchain_nsys.py report.sqlite [--device N] [--top 40]
"""
import argparse
import collections
import re
import sqlite3
import statistics as st


def short(n):
    n = re.sub(r"\(.*$", "", n)
    n = re.sub(r"<.*$", "", n) if not n.startswith("void") else re.sub(r"<.*$", "", n[5:])
    return n.split("::")[-1][:60] if "::" in n and len(n) > 60 else n[:70]


ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--top", type=int, default=40)
a = ap.parse_args()
c = sqlite3.connect(a.report)
names = dict(c.execute("select id, value from StringIds"))
by_corr = collections.defaultdict(list)
for s, e, corr, n, gx, gy, gz, bx in c.execute(
        "select start, end, correlationId, demangledName, gridX, gridY, gridZ, blockX from "
        "CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ? order by start", (a.device,)):
    by_corr[corr].append((s, e, names[n], (gx, gy, gz, bx)))
verify = sorted((ks[0][0], corr) for corr, ks in by_corr.items() if len(ks) >= 500)
steps = [(a0, b0, corr) for (a0, corr), (b0, _) in zip(verify, verify[1:]) if b0 - a0 < 200e6]
vset = {corr for _, _, corr in steps}
allk = sorted((s, e, n, corr) for corr, ks in by_corr.items() for s, e, n, _ in ks)
tot = collections.defaultdict(float)
cnt = collections.Counter()
for a0, b0, corr in steps:
    for s, e, n, cc in allk:
        if a0 <= s < b0:
            k = ("G " if cc in vset else "  ") + short(n)
            tot[k] += (e - s) / 1e3
            cnt[k] += 1
ns = len(steps)
period = st.mean(b - a for a, b, _ in steps) / 1e6
print(f"{ns} steps, period {period:.2f} ms; G = inside the verify graph; us per step (sum of durations)")
for k, v in sorted(tot.items(), key=lambda x: -x[1])[:a.top]:
    print(f"  {v / ns:8.1f} us  x{cnt[k] / ns:6.1f}  mean {v / cnt[k]:7.2f}  {k}")
# median layer chain: verify-graph kernels between consecutive layer_kernel launches
chains = collections.defaultdict(list)
for _, _, corr in steps:
    ks = by_corr[corr]
    idx = [i for i, k in enumerate(ks) if "layer_kernel" in k[2]]
    for i0, i1 in zip(idx[20:40], idx[21:41]):
        seq = ks[i0:i1]
        for j, (s, e, n, g) in enumerate(seq):
            gap = (s - seq[j - 1][1]) / 1e3 if j else 0.0
            chains[j].append((short(n), g, (e - s) / 1e3, gap))
print("\nmedian chain from one layer_kernel to the next (layers 20-40):")
total = 0.0
for j in sorted(chains):
    v = chains[j]
    name = collections.Counter(x[0] for x in v).most_common(1)[0][0]
    g = collections.Counter(x[1] for x in v).most_common(1)[0][0]
    d = st.median(x[2] for x in v)
    gp = st.median(x[3] for x in v)
    total += d + gp
    print(f"  {j:3d} {d:7.2f} us (gap {gp:4.1f})  grid {g[0]}x{g[1]}x{g[2]} blk {g[3]}  {name}")
print(f"  chain total {total:.1f} us")
