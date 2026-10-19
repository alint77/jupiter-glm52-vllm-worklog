"""Eviction policies a GPU kernel can afford, on the top-k dumps (rank 0).

One hot buffer per anchor layer shared by the batch (N rows), a direct
slot -> row table. Per step and layer: the union of the batch's selected slots;
this step's new tokens read reserved rows (never cached; their slots are
invalidated, since a KV write changes the slot's content). Hits first (stamp
the row), then misses take victims, never a row stamped this step:
  lru     exact LRU (needs a sort per step on the GPU)
  fifo    ring hand, skip rows stamped this step
  clock   ring hand with a reference bit set by hits (second chance)
  lru+new exact LRU where the step's new tokens are also inserted (upper bound
          for writing new tokens into real rows instead of reserved ones)

  sim_policy_nb.py <dump dir> [--rows 8192,16384,32768] [--out x.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sim_lru_nb import Pool, steps  # noqa: E402

NSLOTS = 262144


@numba.njit(cache=True)
def ring_step(where, slot_of, stamp, ref, hand, keys, new, step, n, clock):
    for s in new:
        r = where[s]
        if r >= 0:
            where[s] = -1
            slot_of[r] = -1
            stamp[r] = -1
    miss = 0
    for s in keys:
        r = where[s]
        if r >= 0:
            stamp[r] = step
            ref[r] = 1
    for s in keys:
        if where[s] >= 0:
            continue
        miss += 1
        while True:
            c = hand[0]
            hand[0] = (c + 1) % n
            if stamp[c] == step:
                continue
            if clock and ref[c]:
                ref[c] = 0
                continue
            break
        old = slot_of[c]
        if old >= 0:
            where[old] = -1
        where[s] = c
        slot_of[c] = s
        stamp[c] = step
        ref[c] = 0
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--rows", default="8192,16384,32768")
    ap.add_argument("--out")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    caps = [int(c) for c in a.rows.split(",")]
    nl = 21
    pols = ["lru", "fifo", "clock", "lru+new"]
    st = {}
    for c in caps:
        for p in ("fifo", "clock"):
            st[(p, c)] = [(np.full(NSLOTS, -1, np.int32), np.full(c, -1, np.int32),
                           np.full(c, -1, np.int64), np.zeros(c, np.int8),
                           np.zeros(1, np.int64)) for _ in range(nl)]
    lru = {c: Pool(2 * nl, NSLOTS) for c in caps}  # b = li (lru), nl + li (lru+new)
    tot = {}
    nstep = 0
    for e, slots, new, rows in steps(a.dump):
        if a.max_steps and nstep >= a.max_steps:
            break
        if any(r.startswith("_warmup") for r in e["req_ids"]):
            continue
        nstep += 1
        t = e["t"]
        bk = "M<=4" if t <= 4 else "M<=8" if t <= 8 else "M<=12" if t <= 12 else "M<=16"
        nw = np.unique(new[new >= 0]).astype(np.int32)
        nwset = set(nw.tolist())
        for li in range(nl):
            u = np.unique(slots[li][: t])
            u = u[u >= 0].astype(np.int32)
            keys = u[~np.isin(u, nw)]
            d = tot.setdefault(bk, {"steps": 0, "calls": 0, "rows": 0, "miss": {}})
            d["calls"] += 1
            d["rows"] += len(keys)
            for c in caps:
                m = {}
                for p in ("fifo", "clock"):
                    w, so, sp, rf, hd = st[(p, c)][li]
                    m[p] = ring_step(w, so, sp, rf, hd, keys, nw, nstep, c, p == "clock")
                pl = lru[c]
                # lru: new slots invalidated (clear-by-reinsert is not possible
                # in the list, so drop them by touching nothing; reads of a new
                # slot never reach the buffer)
                m["lru"] = pl.access(li, keys, c)
                pl.access(nl + li, nw, c, False)
                m["lru+new"] = pl.access(nl + li, keys, c)
                for p in pols:
                    d["miss"][(p, c)] = d["miss"].get((p, c), 0) + m[p]
        tot[bk]["steps"] += 1
        if nstep % 2000 == 0:
            print(f"step {nstep}", file=sys.stderr, flush=True)
    out = {}
    print(f"{'bucket':6s} {'rows':>6s} {'policy':8s} " + " ".join(f"{c:>13d}" for c in caps))
    for bk, d in sorted(tot.items()):
        rpc = d["rows"] / d["calls"]
        for p in pols:
            cells = []
            for c in caps:
                ms = d["miss"][(p, c)]
                cells.append(f"{100 * ms / d['rows']:5.2f}% {ms / d['calls']:6.1f}")
                out.setdefault(bk, {}).setdefault(p, {})[c] = {
                    "miss_rate": ms / d["rows"], "miss_per_call": ms / d["calls"]}
            print(f"{bk:6s} {rpc:6.0f} {p:8s} " + " ".join(cells))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
