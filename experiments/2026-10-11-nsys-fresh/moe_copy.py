"""Prefill MoE layer span vs overlap with memcpys (cold prefetch over C2C) on
the same device: per layer, span, us of copy overlapping it, copy bytes.

    moe_copy.py report.sqlite [--device N]
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
cps = c.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY "
                "where deviceId=? order by start", (dev,)).fetchall()
layers, cur = [], None
for s, e, dn in rows:
    n = names[dn]
    if "tiered_prefill::" not in n:
        continue
    if "route_kernel" in n:
        cur = [s, e]
        layers.append(cur)
    elif cur:
        cur[1] = max(cur[1], e)
big = [(s, e, b) for s, e, b, k in cps if b >= 1 << 20]
print(f"copies >= 1 MiB: {len(big)}, {sum(b for *_, b in big)/2**30:.1f} GiB, "
      f"rate median {st.median(b/(e-s) for s, e, b in big):.0f} GB/s" if big else "no big copies")
out = []
for s, e in layers:
    ov = sum(max(0, min(e, ce) - max(s, cs)) for cs, ce, _ in big)
    out.append(((e - s) / 1e3, ov / 1e3))
lo = [sp for sp, ov in out if ov < 0.1 * sp]
hi = [sp for sp, ov in out if ov >= 0.5 * sp]
print(f"layers {len(out)}: span median {st.median(sp for sp, _ in out):.0f} us; "
      f"copy-free ({len(lo)}) {st.median(lo) if lo else float('nan'):.0f} us; "
      f">=50% overlapped ({len(hi)}) {st.median(hi) if hi else float('nan'):.0f} us")
