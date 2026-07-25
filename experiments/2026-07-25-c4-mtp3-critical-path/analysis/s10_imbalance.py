"""Is EP rank skew persistent (statically fixable) or per-layer random?"""
import collections, statistics as stat
from tracelib import prep_depth, steps, gpu_ops
from s7_layers import layer_table

depth = 3
# per rank: list over steps of list over layers of the MoE span
data = {}
for rank in range(4):
    ev = prep_depth(rank, depth)
    per_step = []
    for a, b in steps(ev):
        rows = layer_table(ev, a, b)
        if len(rows) == 75:
            per_step.append([r["span"] / 1000 for r in rows])  # ms
    data[rank] = per_step

nst = min(len(v) for v in data.values())
print(f"steps usable: {nst}")

# total routed span per rank per step
print("\nrouted MoE span per rank (ms/step):")
for r in range(4):
    tots = [sum(s) for s in data[r][:nst]]
    print(f"  rank{r}: " + " ".join(f"{t:7.3f}" for t in tots) + f"   mean={stat.mean(tots):7.3f}")

# per-layer imbalance
print("\n=== per-layer imbalance (mean over steps) ===")
imb_static = 0.0   # slowest-rank excess if the SAME rank were always slowest per layer
imb_actual = 0.0
winner = collections.Counter()
layer_rank_mean = []
for ly in range(75):
    per_rank_over_steps = [[data[r][s][ly] for s in range(nst)] for r in range(4)]
    means = [stat.mean(x) for x in per_rank_over_steps]
    layer_rank_mean.append(means)
    winner[means.index(max(means))] += 1
    # actual: mean over steps of (max_rank - mean_rank)
    per_step_excess = []
    for s in range(nst):
        vals = [data[r][s][ly] for r in range(4)]
        per_step_excess.append(max(vals) - stat.mean(vals))
    imb_actual += stat.mean(per_step_excess)
    imb_static += max(means) - stat.mean(means)

print(f"  actual per-step imbalance (max-mean, summed over layers) : {imb_actual:6.3f} ms/step")
print(f"  imbalance explained by persistent per-layer rank ordering : {imb_static:6.3f} ms/step "
      f"({100*imb_static/imb_actual:.0f}%)")
print(f"  residual (step-to-step routing variation)                : {imb_actual-imb_static:6.3f} ms/step "
      f"({100*(imb_actual-imb_static)/imb_actual:.0f}%)")
print(f"\n  which rank is slowest, per layer: {dict(winner)}")

# how concentrated: top-N layers contributing to imbalance
contrib = sorted(((max(m) - stat.mean(m), i) for i, m in enumerate(layer_rank_mean)), reverse=True)
top = contrib[:15]
print(f"\n  top-15 imbalanced layers contribute {sum(c for c, _ in top):.3f} ms "
      f"({100*sum(c for c,_ in top)/imb_static:.0f}% of persistent imbalance):")
for c, i in top:
    m = layer_rank_mean[i]
    print(f"    layer {i:2d}: excess={c:6.3f} ms  ranks=" + " ".join(f"{v:6.3f}" for v in m)
          + f"  spread={max(m)-min(m):6.3f}")

# rank totals: is one rank globally slow?
print("\n=== global per-rank routed totals ===")
tot = [stat.mean([sum(data[r][s]) for s in range(nst)]) for r in range(4)]
print("  " + " ".join(f"rank{r}={tot[r]:7.3f}" for r in range(4))
      + f"   spread={max(tot)-min(tot):.3f} ms ({100*(max(tot)-min(tot))/stat.mean(tot):.1f}%)")
