"""Exposed cold-prefetch copy time per prefill step: per MoE layer, idle time
on the device right before its route_kernel that ends when the layer's slot
copies finish (the MoE waited on the copy), plus copy bandwidth stats.

    copy_wait.py report.sqlite [--device N]

Assumes the step's H2D copies >= 1 MiB come in layer order, the same count per
layer (the 2-slot prefetch stages each MoE layer's cold experts once).
"""
import argparse, sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("--device", type=int, default=0)
a = ap.parse_args()
c = sqlite3.connect(a.report)
names = dict(c.execute("select id, value from StringIds"))
ks = [(s, e, names[n]) for s, e, n in c.execute(
    "select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start", (a.device,))]
cps = c.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? and copyKind=1 "
                "and bytes>=1048576 order by start", (a.device,)).fetchall()
routes = [i for i, k in enumerate(ks) if k[2] == "route_kernel"]
per = len(cps) // len(routes)
assert per * len(routes) == len(cps), (len(cps), len(routes))
wait = 0; waits = []
busy_end = 0
ends = []
for i, (s, e, n) in enumerate(ks):
    ends.append(busy_end)
    busy_end = max(busy_end, e)
for li, ri in enumerate(routes):
    rs = ks[ri][0]
    cend = max(ce for _, ce, _ in cps[li * per:(li + 1) * per])
    prev = ends[ri]                       # latest end of any earlier kernel
    w = max(0, min(rs, cend) - prev) if cend > prev else 0
    waits.append(w); wait += w
span = ks[-1][1] - ks[0][0]
cbytes = sum(b for *_, b in cps); cbusy = sum(ce - cs for cs, ce, _ in cps)
first = cps[0][0] - ks[0][0]
print(f"step span {span/1e6:6.1f} ms | copies {cbytes/2**30:.1f} GiB in {cbusy/1e6:5.1f} ms busy "
      f"({cbytes/cbusy:.0f} GB/s), first copy at +{first/1e6:.1f} ms | MoE waited on copy {wait/1e6:5.2f} ms "
      f"over {sum(w > 0 for w in waits)}/{len(waits)} layers (max {max(waits)/1e3:.0f} us)")

slack = []
for li, ri in enumerate(routes):
    cend = max(ce for _, ce, _ in cps[li * per:(li + 1) * per])
    slack.append(ks[ri][0] - cend)
sl = sorted(slack)
print(f"   slack before MoE (route start - its copies done): min {sl[0]/1e3:.0f} us, p10 {sl[len(sl)//10]/1e3:.0f} us, median {sl[len(sl)//2]/1e3:.0f} us")

