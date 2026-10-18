"""Speed of light of every served sliced MoE call (route_fit.py's
an/calls-*.csv) by the distinct hot / cold experts it touched.

Per GPU and call: hot slices h x 5.06 MiB plus the fused bf16 shared-expert
slice (18 MiB) from HBM at 3.6 TB/s, cold slices c x 5.06 MiB over C2C at
421 GB/s (2026-10-04-c2c-roofline ceiling); ideal = the slower stream (they
run concurrently). SOL% = ideal / measured layer_kernel time. Heatmap per M:
cell colour = mean SOL% of the calls in the (hot bin, cold) cell, text = mean
SOL% and call count; the line is where the two streams take equal time.

    sol_heatmap.py OUT.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AN = Path(__file__).resolve().parent / "an"
SLICE = 5.06 * 2**20
SHARED = (1024 * 6144 + 6144 * 512) * 2
HBM, C2C = 3.6e12, 421e9
HBIN = 4
ARMS = {8: ["rt-dec2-2390691"], 16: ["rt-ctx4-2390685", "rt-dec4-2390684", "rt-tlnsys-2390683"]}


def load(arms, m):
    R = np.concatenate([np.loadtxt(AN / f"calls-{a}.csv", delimiter=",", skiprows=1) for a in arms])
    return R[R[:, 2] == m]


def main():
    fig, axes = plt.subplots(1, len(ARMS), figsize=(11 * len(ARMS), 11), squeeze=False)
    for ax, (m, arms) in zip(axes[0], ARMS.items()):
        R = load(arms, m)
        h, c, us = R[:, 4], R[:, 5], R[:, 6]
        t_hbm = (h * SLICE + SHARED) / HBM * 1e6
        t_c2c = c * SLICE / C2C * 1e6
        sol = np.maximum(t_hbm, t_c2c) / us * 100
        hb = (h // HBIN).astype(int)
        hmin, hmax, cmax = hb.min(), hb.max(), int(c.max())
        grid = np.full((cmax + 1, hmax - hmin + 1), np.nan)
        cnt = np.zeros_like(grid, dtype=int)
        for i in range(len(R)):
            y, x = int(c[i]), hb[i] - hmin
            cnt[y, x] += 1
            grid[y, x] = sol[i] if cnt[y, x] == 1 else grid[y, x] + sol[i]
        grid = np.where(cnt > 0, grid / np.maximum(cnt, 1), np.nan)
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=100,
                       extent=(hmin * HBIN, (hmax + 1) * HBIN, -0.5, cmax + 0.5))
        for y in range(cmax + 1):
            for x in range(hmax - hmin + 1):
                if cnt[y, x]:
                    ax.text((x + hmin + 0.5) * HBIN, y, f"{grid[y, x]:.0f}\n{cnt[y, x]}",
                            ha="center", va="center", fontsize=5.5,
                            color="white" if grid[y, x] < 55 else "black")
        hh = np.linspace(hmin * HBIN, (hmax + 1) * HBIN, 50)
        ax.plot(hh, (hh * SLICE + SHARED) / HBM * C2C / SLICE, "r--", lw=1.2,
                label="HBM time = C2C time (above: C2C-bound)")
        ax.set_xlabel(f"distinct hot experts touched (bins of {HBIN})")
        ax.set_ylabel("distinct cold experts touched")
        tot = np.maximum(t_hbm, t_c2c).sum() / us.sum() * 100
        ax.set_title(f"M={m}: {len(R):,} layer_kernel calls ({', '.join(a.split('-')[1] for a in arms)}); "
                     f"time-weighted SOL {tot:.0f}%, median {np.median(sol):.0f}%\n"
                     "cell: mean SOL% / calls", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        fig.colorbar(im, ax=ax, label="mean SOL % (ideal max(HBM, C2C) / measured)")
        c2c_bound = t_c2c > t_hbm
        print(f"M={m}: {len(R)} calls, SOL time-weighted {tot:.1f}%, median {np.median(sol):.1f}%; "
              f"C2C-bound {c2c_bound.mean() * 100:.0f}% of calls (SOL {np.median(sol[c2c_bound]):.0f}%), "
              f"HBM-bound SOL {np.median(sol[~c2c_bound]):.0f}%; "
              f"measured {us.sum() / 1e3:.0f} ms vs ideal {np.maximum(t_hbm, t_c2c).sum() / 1e3:.0f} ms")
        for lo, hi in ((0, 2), (3, 5), (6, 9), (10, 14), (15, 99)):
            s = (c >= lo) & (c <= hi)
            if s.any():
                print(f"   cold {lo:2d}-{hi:2d}: {s.sum():6d} calls, median SOL {np.median(sol[s]):4.0f}%, "
                      f"mean us {us[s].mean():5.0f} vs ideal {np.maximum(t_hbm, t_c2c)[s].mean():5.0f}")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=170)
    print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
