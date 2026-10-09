"""Timeline of one traced call (kdev.py once --n 1 --trace t.pt on a
TD_CTA_TRACE build): per kernel the span from first entry to last exit, and
for the two GEMMs the hot / cold CTA groups' ready (past PDL wait) and exit
spread, all relative to route_prep's first entry, in us.

    tl.py t.pt [--cold-ctas 16]
"""
import argparse

import numpy as np
import torch

NAMES = {50: "route_prep", 53: "act (live)", 63: "act (unrouted)", 54: "finalize",
         0: "w13 CTA", 1: "w2 CTA", 10: "w13 CTA idle", 11: "w2 CTA idle"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    a = ap.parse_args()
    r = torch.load(a.trace).numpy().astype(np.int64)
    ph = r[:, 0] & 0xFF
    blk = (r[:, 0] >> 8) & 0xFF
    n_hot = (r[:, 0] >> 32) & 0xFFFF
    n_cold = (r[:, 0] >> 48) & 0xFFFF
    t0 = r[(ph == 50), 1].min()
    us = lambda x: (x - t0) / 1e3  # noqa: E731
    print(f"hot {n_hot[ph == 0].max()} cold {n_cold[ph == 0].max()}")
    for p in (50, 0, 10, 53, 63, 1, 11, 54):
        m = ph == p
        if not m.any():
            continue
        print(f"{NAMES[p]:16s} n={m.sum():4d} entry {us(r[m, 1].min()):7.1f}..{us(r[m, 1].max()):7.1f}"
              f"  ready {us(r[m, 2].min()):7.1f}..{us(r[m, 2].max()):7.1f}"
              f"  exit {us(r[m, 3].min()):7.1f} / p50 {us(np.median(r[m, 3])):7.1f} / {us(r[m, 3].max()):7.1f}")
    for p in (0, 1):
        m = ph == p
        if not m.any():
            continue
        cold = n_cold[m].max()
        nc = 0 if cold == 0 else (16 if cold == 1 else 24)
        for name, sel in (("hot", blk[m] >= nc), ("cold", blk[m] < nc)):
            if not sel.any():
                continue
            ex = us(r[m][sel, 3])
            rd = us(r[m][sel, 2])
            print(f"  w{'13' if p == 0 else '2'} {name:4s} CTAs {sel.sum():3d}: ready "
                  f"{rd.min():6.1f}..{rd.max():6.1f}, exit {ex.min():6.1f} p10 "
                  f"{np.percentile(ex, 10):6.1f} p50 {np.median(ex):6.1f} p90 "
                  f"{np.percentile(ex, 90):6.1f} max {ex.max():6.1f}")
        for q, name in ((p + 60, "producer"), (p + 70, "consumer")):
            mm = ph == q
            if mm.any():
                print(f"  w{'13' if p == 0 else '2'} {name}: first {us(r[mm, 1]).min():6.1f}"
                      f" p50 {np.median(us(r[mm, 1])):6.1f}; last p50 {np.median(us(r[mm, 3])):6.1f}"
                      f" max {us(r[mm, 3]).max():6.1f}")


if __name__ == "__main__":
    main()
