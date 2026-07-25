"""MoE routed-tier scaling across MTP depth (verify batch = 8 / 12 / 16 tokens)."""
import collections, os, statistics as stat
import tracelib as core
from s7_layers import layer_table

# per-expert W4A16 bytes (H=6144, I=2048, g=128, int4 + bf16 group scales)
H, I, G = 6144, 2048, 128
W13 = (2 * I * H) // 2 + (2 * I) * (H // G) * 2
W2 = (H * I) // 2 + H * (I // G) * 2
print(f"per-expert bytes: w13={W13:,}  w2={W2:,}  total={W13+W2:,}  (ratio {W13/W2:.3f})")

VERIFY = {1: 8, 2: 12, 3: 16}
res = {}
for depth in (1, 2, 3):
    agg = collections.Counter()
    nsteps = 0
    for rank in range(4):
        ev = core.prep_depth(rank, depth)
        st = core.steps(ev)
        for a, b in st:
            rows = layer_table(ev, a, b)
            if len(rows) < 70:
                continue
            nsteps += 1
            for r in rows:
                agg["hot_w13"] += r["hot_w13"]; agg["hot_w2"] += r["hot_w2"]
                agg["cold_w13"] += r["cold_w13"]; agg["cold_w2"] += r["cold_w2"]
                agg["hot_act"] += r["hot_act"]; agg["cold_act"] += r["cold_act"]
                agg["span"] += r["span"]; agg["sum"] += r["marlin_sum"]
                agg["n"] += 1
    for k in agg:
        agg[k] /= nsteps
    res[depth] = (agg, nsteps)
    print(f"\n--- MTP{depth}: verify={VERIFY[depth]} tokens, {nsteps} rank-steps, "
          f"{agg['n']:.0f} routed layers/step")
    hot = (agg["hot_w13"] + agg["hot_act"] + agg["hot_w2"]) / 1000
    cold = (agg["cold_w13"] + agg["cold_act"] + agg["cold_w2"]) / 1000
    print(f"  hot  chain {hot:7.3f} ms  (w13 {agg['hot_w13']/1000:6.3f} act {agg['hot_act']/1000:5.3f} w2 {agg['hot_w2']/1000:6.3f})  w13/w2={agg['hot_w13']/agg['hot_w2']:.2f}")
    print(f"  cold chain {cold:7.3f} ms  (w13 {agg['cold_w13']/1000:6.3f} act {agg['cold_act']/1000:5.3f} w2 {agg['cold_w2']/1000:6.3f})  w13/w2={agg['cold_w13']/agg['cold_w2']:.2f}")
    print(f"  marlin sum {agg['sum']/1000:7.3f} ms   layer-span total {agg['span']/1000:7.3f} ms   overlap saving {100*(1-agg['span']/agg['sum']):.1f}%")

print("\n=== scaling of routed MoE with verify batch ===")
print(f"{'depth':>5} {'tokens':>7} {'marlin_sum':>11} {'span':>9} {'hot':>8} {'cold':>8}")
for d in (1, 2, 3):
    a, _ = res[d]
    hot = (a["hot_w13"] + a["hot_act"] + a["hot_w2"]) / 1000
    cold = (a["cold_w13"] + a["cold_act"] + a["cold_w2"]) / 1000
    print(f"{d:5d} {VERIFY[d]:7d} {a['sum']/1000:11.3f} {a['span']/1000:9.3f} {hot:8.3f} {cold:8.3f}")

# linear fit span = fixed + slope * tokens
xs = [VERIFY[d] for d in (1, 2, 3)]
for label, key in (("marlin_sum", "sum"), ("layer span", "span")):
    ys = [res[d][0][key] / 1000 for d in (1, 2, 3)]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    icept = (sy - slope * sx) / n
    print(f"\n{label}: {icept:.3f} ms fixed + {slope:.4f} ms/token   "
          f"(at 16 tok: fixed share {100*icept/(icept+slope*16):.0f}%)")
