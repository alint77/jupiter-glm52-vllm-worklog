"""Consolidated tier model: derive activated experts, effective bandwidth, rebalance optimum."""
H, I, G = 6144, 2048, 128
W13 = (2 * I * H) // 2 + (2 * I) * (H // G) * 2
W2 = (H * I) // 2 + H * (I // G) * 2
E = W13 + W2
L = 75
C2C = 421e9
HBM = 3.5e12

# measured (MTP3, mean of 20 rank-steps, ms/step)
m = dict(hot_w13=16.673, hot_w2=7.336, hot_act=0.513,
         cold_w13=6.580, cold_w2=11.720, cold_act=2.602,
         span=25.716, cum=42.308)

print(f"per-expert bytes: w13={W13/1e6:.3f} MB  w2={W2/1e6:.3f} MB  total={E/1e6:.3f} MB\n")

# cold w13 is the least-contended measurement (launched first, owns the GPU)
n_c = m["cold_w13"] * 1e-3 * C2C / (W13 * L)
bw_c = n_c * W13 * L / (m["cold_w13"] * 1e-3)
print(f"COLD tier (anchored on w13, the uncontended launch):")
print(f"  cold w13 = {m['cold_w13']:.3f} ms -> {n_c:.2f} activated cold experts/layer at the {C2C/1e9:.0f} GB/s C2C roof")
print(f"  implied effective C2C bandwidth = {bw_c/1e9:.0f} GB/s  ({100*bw_c/C2C:.0f}% of spec)")
cold_solo = m["cold_w13"] * 1.5 + m["cold_act"] * 0.2
print(f"  cold w2 at the same rate would be {m['cold_w13']/2:.3f} ms; measured {m['cold_w2']:.3f} ms "
      f"-> {m['cold_w2']/(m['cold_w13']/2):.1f}x contention dilation")
print(f"  => cold SOLO cost ~= {cold_solo:.2f} ms (vs {m['cold_w13']+m['cold_w2']+m['cold_act']:.2f} ms apparent)\n")

print("HOT tier (assumes D distinct activated experts/layer/rank):")
print(f"  {'D':>5} {'n_hot':>7} {'eff BW':>10} {'% HBM SOL':>10}")
for D in (18, 20, 22, 25.3):
    n_h = D - n_c
    t_h = (m["hot_w13"] + m["hot_w2"] + m["hot_act"]) * 1e-3
    bw_h = n_h * E * L / t_h
    print(f"  {D:5.1f} {n_h:7.2f} {bw_h/1e12:9.2f}T {100*bw_h/HBM:9.0f}%")

print(f"\n  hot chain = {m['hot_w13']+m['hot_w2']+m['hot_act']:.2f} ms; layer-span total = {m['span']:.2f} ms")
print(f"  => overlap is already within {m['span']-(m['hot_w13']+m['hot_w2']+m['hot_act']):.2f} ms of perfect "
      f"(span == max(hot,cold)); the HOT tier is the routed critical path\n")

print("REBALANCE optimum (minimise max(t_hot, t_cold) by moving experts hot->cold):")
for D in (20, 22, 25.3):
    n_h0 = D - n_c
    t_h0 = (m["hot_w13"] + m["hot_w2"] + m["hot_act"])
    per_hot = t_h0 / n_h0            # ms per activated hot expert (all layers)
    per_cold = m["cold_w13"] * 1.5 / n_c
    n_c_opt = D * per_hot / (per_hot + per_cold)
    n_h_opt = D - n_c_opt
    t_opt = n_h_opt * per_hot
    print(f"  D={D:5.1f}: now hot={n_h0:5.2f}/cold={n_c:.2f} -> span {t_h0:5.2f} ms | "
          f"opt hot={n_h_opt:5.2f}/cold={n_c_opt:4.2f} -> span {t_opt:5.2f} ms  "
          f"(save {t_h0-t_opt:4.2f} ms = {100*(t_h0-t_opt)/62.43:.1f}% of step)")
    print(f"           per-expert cost: hot {per_hot*1000:.0f} us, cold {per_cold*1000:.0f} us "
          f"(ratio {per_cold/per_hot:.2f})")

print("\nVERIFY-BATCH SCALING (measured across MTP depth, 4 sequences):")
pts = [(8, 17.257), (12, 22.142), (16, 25.716)]
for i in range(1, len(pts)):
    dt = (pts[i][1] - pts[i-1][1]) / (pts[i][0] - pts[i-1][0])
    print(f"  {pts[i-1][0]:2d}->{pts[i][0]:2d} tokens: {dt:.3f} ms of routed span per extra token")
print("  marginal cost is DECLINING -> expert reuse is starting to amortise, but only weakly")
