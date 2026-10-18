"""Per-call sliced MoE time vs the experts it touched, on served traffic.

Joins a route-record arm (worktrees/pf5rt: route/route.r0.{start,stop}<n>.pt,
the routed expert ids of every verify step, rank 0; routing is identical on
every TP rank) with the arm's nsys window n: the window's verify steps (CUDA
graph replays with >= 500 kernels, device 0) are the ring's steps
ctr_start+1 .. ctr_stop, and each step's 75 layer_kernel calls are the 75
routed layers in order. Per call: distinct hot h and cold c experts touched
(the server's own hot/cold maps), layer_kernel us. Fits us = a + b*h + g*c
per step size M and reports how much of the per-layer spread it explains.

    route_fit.py <arm out dir> [--device 0] [--csv calls.csv]
"""
import argparse
import collections
import sqlite3
from pathlib import Path

import numpy as np
import torch


def verify_steps(db, device):
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    ks = c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL "
                   "where deviceId=? order by start", (device,)).fetchall()
    per = collections.Counter(k[3] for k in ks)
    steps = collections.defaultdict(list)
    for s, e, n, corr in ks:
        if per[corr] >= 500 and names[n] == "layer_kernel":
            steps[corr].append((s, (e - s) / 1e3))
    return [[d for _, d in sorted(v)] for _, v in sorted(steps.items(), key=lambda kv: min(kv[1]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--csv")
    a = ap.parse_args()
    d = Path(a.dir)
    rows = []
    for db in sorted(d.glob("nsys.*.sqlite"), key=lambda p: int(p.name.split(".")[1])):
        n = int(db.name.split(".")[1])
        st, sp = (d / "route" / f"route.r0.{t}{n}.pt" for t in ("start", "stop"))
        if not (st.exists() and sp.exists()):
            print(f"window {n}: no route dumps")
            continue
        r0, r1 = torch.load(st), torch.load(sp)
        ids, m, steps = r1["ids"].numpy(), r1["m"].numpy(), verify_steps(db, a.device)
        rec = list(range(r0["ctr"] + 1, r1["ctr"] + 1))
        layers = sorted(li for li in r1["maps"] if li < 78)
        hot = np.stack([r1["maps"][li][0].numpy() for li in layers])
        cold = np.stack([r1["maps"][li][1].numpy() for li in layers])
        ok = [s for s in steps if len(s) == len(layers)]
        print(f"window {n}: {len(rec)} recorded steps, {len(steps)} nsys verify steps "
              f"({len(ok)} with {len(layers)} calls)")
        if len(rec) != len(steps) or len(rec) > len(ids):
            print("  step counts differ: skipped")
            continue
        for k, times in zip(rec, steps):
            if len(times) != len(layers):
                continue
            slot = k % len(ids)
            for j, li in enumerate(layers):
                e = ids[slot, li]
                e = np.unique(e[e != 255])
                rows.append((n, k, int(m[slot]), j, int(hot[j, e].sum()), int(cold[j, e].sum()), times[j]))
    if not rows:
        return
    R = np.array(rows, dtype=float)
    if a.csv:
        np.savetxt(a.csv, R, fmt="%g", delimiter=",", header="win,step,M,layer,hot,cold,us", comments="")
    for M in sorted(set(R[:, 2].astype(int))):
        X = R[R[:, 2] == M]
        if len(X) < 300:
            continue
        A = np.c_[np.ones(len(X)), X[:, 4], X[:, 5]]
        coef, *_ = np.linalg.lstsq(A, X[:, 6], rcond=None)
        pred = A @ coef
        res = X[:, 6] - pred
        r2 = 1 - res.var() / X[:, 6].var()
        L = X[:, 3].astype(int)
        lm = np.array([X[L == j, 6].mean() for j in range(75)])
        pm = np.array([pred[L == j].mean() for j in range(75)])
        r2l = 1 - ((lm - pm) ** 2).mean() / lm.var()
        print(f"M={M}: {len(X) // 75} steps, us = {coef[0]:.1f} + {coef[1]:.2f} h + {coef[2]:.2f} c; "
              f"R2 per call {r2:.2f}, per-layer means {r2l:.2f}; resid sd {res.std():.1f} us; "
              f"mean h {X[:, 4].mean():.1f} c {X[:, 5].mean():.1f}")


if __name__ == "__main__":
    main()
