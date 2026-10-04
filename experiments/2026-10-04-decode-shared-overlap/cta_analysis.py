"""Per-CTA decode MoE trace from serving (TD_CTA_TRACE build, dumped at
/stop_profile): per w13 / w2 launch, the hot / cold expert counts, when the hot
and cold CTAs finish, which tier ends the kernel, how long only cold CTAs run,
idle SM time, and the bandwidth each tier gets.
Usage: cta_analysis.py <dir with rank*-stop.pt>"""
import statistics as st
import sys
from pathlib import Path

import torch

GRID = 132
# bytes per expert per phase (INT4 + bf16 group-32 scales): w13 4096 x 6144,
# w2 6144 x 2048
EXPERT_BYTES = {0: 4096 * 6144 // 2 + 6144 // 32 * 4096 * 2,
                1: 6144 * 2048 // 2 + 2048 // 32 * 6144 * 2}


def cold_ctas(n_hot, n_cold):
    if n_cold == 0:
        return 0
    want = 16 if n_cold == 1 else 24
    return want if n_hot == 0 else min(want, GRID - 1)


def launches(recs):
    rows = []
    for w0, t0, t1, t2 in recs.tolist():
        ph = w0 & 0xFF
        if ph >= 100:
            continue
        rows.append({"ph": ph % 10, "empty": ph >= 10, "blk": (w0 >> 8) & 0xFF,
                     "sm": (w0 >> 16) & 0xFFFF, "nh": (w0 >> 32) & 0xFFFF,
                     "nc": (w0 >> 48) & 0xFFFF, "t0": t0, "t1": t1, "t2": t2})
    out = {0: [], 1: []}
    for ph in (0, 1):
        rs = sorted((r for r in rows if r["ph"] == ph), key=lambda r: r["t0"])
        cur = []
        for r in rs:
            if cur and r["t0"] - cur[-1]["t0"] > 20_000:
                out[ph].append(cur)
                cur = []
            cur.append(r)
        if cur:
            out[ph].append(cur)
    return {ph: [L for L in ls if len(L) == GRID] for ph, ls in out.items()}


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def report(path):
    recs = torch.load(path)
    ls = launches(recs)
    print(f"== {path.name}: {len(ls[0])} w13 / {len(ls[1])} w2 launches")
    for ph, name in ((0, "w13"), (1, "w2")):
        agg = {k: [] for k in ("dur", "hot_end", "cold_end", "cold_tail", "idle_frac",
                               "hot_bw", "cold_bw", "nh", "nc", "cold_last", "wait")}
        for L in ls[ph]:
            nh, nc = L[0]["nh"], L[0]["nc"]
            cc = cold_ctas(nh, nc)
            ready = min(r["t1"] for r in L if r["t1"])
            start = min(r["t0"] for r in L)
            end = max(r["t2"] for r in L)
            hot = [r for r in L if r["blk"] >= cc and not r["empty"]]
            cold = [r for r in L if r["blk"] < cc and not r["empty"]]
            agg["nh"].append(nh)
            agg["nc"].append(nc)
            agg["dur"].append((end - ready) / 1e3)
            agg["wait"].append((ready - start) / 1e3)
            if hot:
                he = max(r["t2"] for r in hot)
                agg["hot_end"].append((he - ready) / 1e3)
                agg["hot_bw"].append(nh * EXPERT_BYTES[ph] / (he - ready))  # GB/s
            if cold:
                ce = max(r["t2"] for r in cold)
                agg["cold_end"].append((ce - ready) / 1e3)
                agg["cold_bw"].append(nc * EXPERT_BYTES[ph] / (ce - ready))
                if hot:
                    agg["cold_tail"].append(max(0, ce - max(r["t2"] for r in hot)) / 1e3)
                    agg["cold_last"].append(ce > max(r["t2"] for r in hot))
            busy = sum(max(0, min(r["t2"], end) - max(r["t1"] or ready, ready)) for r in L)
            agg["idle_frac"].append(1 - busy / (GRID * (end - ready)))
        n = len(agg["dur"])
        print(f"  {name}: launches {n}, hot experts median {st.median(agg['nh'])} "
              f"(p10-p90 {q(agg['nh'], .1)}-{q(agg['nh'], .9)}), cold {st.median(agg['nc'])} "
              f"({q(agg['nc'], .1)}-{q(agg['nc'], .9)}); PDL wait before ready {st.median(agg['wait']):.1f} us")
        print(f"     kernel (ready -> last CTA) median {st.median(agg['dur']):.1f} us; "
              f"hot CTAs done at {st.median(agg['hot_end']):.1f} us "
              f"({st.median(agg['hot_bw']):.0f} GB/s HBM)"
              + (f"; cold done at {st.median(agg['cold_end']):.1f} us "
                 f"({st.median(agg['cold_bw']):.0f} GB/s C2C); cold ends last in "
                 f"{100 * sum(agg['cold_last']) / max(1, len(agg['cold_last'])):.0f}% of launches "
                 f"with both, cold-only tail median {st.median(agg['cold_tail']):.1f} us "
                 f"(p90 {q(agg['cold_tail'], .9):.1f})" if agg["cold_end"] else ""))
        print(f"     SM-time idle inside the kernel window: median {100 * st.median(agg['idle_frac']):.0f}%")


if __name__ == "__main__":
  for p in sorted(Path(sys.argv[1]).glob("rank*-stop.pt")):
    report(p)
