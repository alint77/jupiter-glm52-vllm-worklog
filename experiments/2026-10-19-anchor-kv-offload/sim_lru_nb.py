"""LRU hot-buffer simulation over a VLLM_ANCHOR_TOPK_DUMP dump (rank 0), numba.

Per step and anchor layer a request reads the union over its query tokens of
their converted top-k slots (>= 0, rank-local physical slots). The step's new
tokens are written into the buffers by the KV write (inserted as most recent,
not counted). Policies, capacity in rows per rank:
  req:    one LRU per (request, layer), dropped when the request leaves the
          batch (HiSparse's layout; a new turn of the same conversation starts
          cold even though its prefix blocks are shared)
  global: one LRU per layer shared by the batch (capacity x4 = same total as
          4 per-request buffers); prefix-shared blocks survive across turns
A miss is a row the anchor would read from Grace on its critical path. Reports
per M bucket: miss rate, misses per anchor call (summed over the batch) mean
and p50/p90/p99, and the same excluding each request's first 8 steps (cold).

  sim_lru_nb.py <dump dir> [--caps 512,1024,2048,4096,8192] [--out x.json]
"""

import argparse
import json
import os

import numba
import numpy as np

COLD = 8


@numba.njit(cache=True)
def access(prev, nxt, inb, hd, tl, sz, b, keys, cap, count):
    miss = 0
    for k in keys:
        if inb[b, k]:
            if hd[b] == k:
                continue
            p, n = prev[b, k], nxt[b, k]
            nxt[b, p] = n
            if n >= 0:
                prev[b, n] = p
            else:
                tl[b] = p
        else:
            if count:
                miss += 1
            inb[b, k] = 1
            sz[b] += 1
        prev[b, k] = -1
        nxt[b, k] = hd[b]
        if hd[b] >= 0:
            prev[b, hd[b]] = k
        hd[b] = k
        if tl[b] < 0:
            tl[b] = k
        while sz[b] > cap:
            t = tl[b]
            inb[b, t] = 0
            tl[b] = prev[b, t]
            nxt[b, tl[b]] = -1
            sz[b] -= 1
    return miss


@numba.njit(cache=True)
def clear(prev, nxt, inb, hd, tl, sz, b):
    k = hd[b]
    while k >= 0:
        inb[b, k] = 0
        k = nxt[b, k]
    hd[b] = -1
    tl[b] = -1
    sz[b] = 0


class Pool:
    def __init__(self, nbuf, nslots):
        self.prev = np.full((nbuf, nslots), -1, np.int32)
        self.nxt = np.full((nbuf, nslots), -1, np.int32)
        self.inb = np.zeros((nbuf, nslots), np.int8)
        self.hd = np.full(nbuf, -1, np.int32)
        self.tl = np.full(nbuf, -1, np.int32)
        self.sz = np.zeros(nbuf, np.int32)

    def access(self, b, keys, cap, count=True):
        return access(self.prev, self.nxt, self.inb, self.hd, self.tl, self.sz,
                      b, keys, cap, count)

    def clear(self, b):
        clear(self.prev, self.nxt, self.inb, self.hd, self.tl, self.sz, b)


def steps(path):
    with open(os.path.join(path, "index.jsonl")) as f:
        idx = [json.loads(line) for line in f]
    data = np.memmap(os.path.join(path, "topk.bin"), dtype=np.int32, mode="r")
    for e in idx:
        L, t, w = len(e["layers"]), e["t"], e["w"]
        o = e["off"] // 4
        slots = np.asarray(data[o : o + L * t * w]).reshape(L, t, w)
        o += L * t * w
        new = np.asarray(data[o : o + t])
        rows = np.asarray(data[o + t : o + 2 * t])
        yield e, slots, new, rows


def bucket(t):
    return "M<=4" if t <= 4 else "M<=8" if t <= 8 else "M<=12" if t <= 12 else "M<=16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--caps", default="512,1024,2048,4096,8192")
    ap.add_argument("--nslots", type=int, default=262144)
    ap.add_argument("--out")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    caps = [int(c) for c in a.caps.split(",")]
    nl = 21
    maxreq = 16
    pools = {c: Pool(maxreq * nl + nl, a.nslots) for c in caps}
    slot_of = {}  # req id -> buffer base (per-request buffers b = base*nl + li)
    free = list(range(maxreq))
    age = {}
    # per (bucket, policy, cap, warm): list of misses per anchor call
    calls = {}
    rows_read = {}
    nsteps = {}
    for n, (e, slots, new, rows) in enumerate(steps(a.dump)):
        if a.max_steps and n >= a.max_steps:
            break
        ids = e["req_ids"]
        if any(r.startswith("_warmup") for r in ids):
            continue
        cur = set(ids)
        for gone in [r for r in slot_of if r not in cur]:
            base = slot_of.pop(gone)
            for c in caps:
                for li in range(nl):
                    pools[c].clear(base * nl + li)
            free.append(base)
            age.pop(gone, None)
        for r in ids:
            if r not in slot_of:
                slot_of[r] = free.pop()
            age[r] = age.get(r, -1) + 1
        t = e["t"]
        bk = bucket(t)
        nsteps[bk] = nsteps.get(bk, 0) + 1
        warm_all = all(age[r] >= COLD for r in ids)
        per_row = []
        for r in np.unique(rows[rows >= 0]):
            if r >= len(ids):
                continue
            sel = rows == r
            nw = new[sel]
            per_row.append((slot_of[ids[r]], sel, np.unique(nw[nw >= 0]).astype(np.int32)))
        new_all = np.unique(new[new >= 0]).astype(np.int32)
        for li in range(nl):
            s = slots[li]
            m_req = {c: 0 for c in caps}
            nread = 0
            union = []
            for base, sel, nw in per_row:
                u = np.unique(s[sel])
                u = u[u >= 0].astype(np.int32)
                union.append(u)
                nread += len(u)
                b = base * nl + li
                for c in caps:
                    pools[c].access(b, nw, c, False)
                    m_req[c] += pools[c].access(b, u, c)
            ug = np.unique(np.concatenate(union)) if union else np.zeros(0, np.int32)
            for c in caps:
                gb = maxreq * nl + li
                pools[c].access(gb, new_all, 4 * c, False)
                mg = pools[c].access(gb, ug.astype(np.int32), 4 * c)
                for warm in (False, True):
                    if warm and not warm_all:
                        continue
                    calls.setdefault((bk, "req", c, warm), []).append(m_req[c])
                    calls.setdefault((bk, "global", c, warm), []).append(mg)
            for warm in (False, True):
                if warm and not warm_all:
                    continue
                rows_read[(bk, warm)] = rows_read.get((bk, warm), 0) + nread
    out = {"steps": nsteps, "results": []}
    for (bk, pol, c, warm), v in sorted(calls.items()):
        v = np.asarray(v)
        out["results"].append({
            "bucket": bk, "policy": pol, "cap": c, "warm": warm, "calls": len(v),
            "rows_per_call": rows_read[(bk, warm)] / len(v),
            "miss_rate": float(v.sum() / max(rows_read[(bk, warm)], 1)),
            "miss_mean": float(v.mean()),
            "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)),
        })
    print(f"steps {nsteps}")
    print(f"{'bucket':6s} {'policy':6s} {'cap':>5s} {'warm':>4s} {'calls':>7s} {'rows':>6s} "
          f"{'miss%':>6s} {'mean':>7s} {'p50':>6s} {'p90':>6s} {'p99':>6s}")
    for r in out["results"]:
        print(f"{r['bucket']:6s} {r['policy']:6s} {r['cap']:5d} {str(r['warm'])[0]:>4s} "
              f"{r['calls']:7d} {r['rows_per_call']:6.0f} {100 * r['miss_rate']:6.2f} "
              f"{r['miss_mean']:7.1f} {r['p50']:6.0f} {r['p90']:6.0f} {r['p99']:6.0f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
