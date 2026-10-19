"""LRU hot-buffer simulation over a VLLM_ANCHOR_TOPK_DUMP dump (rank 0).

Per step and anchor layer, the rows a decode step reads are the union over its
query tokens of their converted top-k slots (>= 0). A row is a hit if it is in
the hot buffer; this step's own new tokens are written into the buffer by the
KV write, so they always hit. Policies (capacity in rows per rank):
  req:    one LRU per (request, layer), HiSparse's layout
  global: one LRU per layer shared by every request (shared prefix rows once)
Requests whose rows all fit (context share on this rank <= capacity) never
miss in a per-request buffer once filled; first touch of a row counts as a
miss unless --warm (a resident request demoted at that point arrives with an
empty buffer; HiSparse fills on demand too).

  sim_lru.py <dump dir> [--caps 512,1024,2048,4096] [--out results.json]
"""

import argparse
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np


class LRU:
    __slots__ = ("cap", "d")

    def __init__(self, cap):
        self.cap = cap
        self.d = OrderedDict()

    def access(self, keys, pinned):
        """Touch keys (iterable of ints); return misses. pinned keys always hit
        (inserted without counting)."""
        d = self.d
        miss = 0
        for k in keys:
            if k in d:
                d.move_to_end(k)
            else:
                if k not in pinned:
                    miss += 1
                d[k] = None
        while len(d) > self.cap:
            d.popitem(last=False)
        return miss


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--caps", default="512,1024,2048,4096,8192")
    ap.add_argument("--warmup", type=int, default=0, help="steps excluded from counts")
    ap.add_argument("--out")
    a = ap.parse_args()
    caps = [int(c) for c in a.caps.split(",")]
    req = {c: defaultdict(lambda c=c: LRU(c)) for c in caps}
    glob = {c: defaultdict(lambda c=c: LRU(c)) for c in caps}
    # totals per (M bucket) -> [rows read, misses req per cap, misses global per cap]
    tot = defaultdict(lambda: {"steps": 0, "rows": 0, "new": 0,
                               "req": defaultdict(int), "glob": defaultdict(int)})
    per_layer = defaultdict(lambda: {"rows": 0, "req": defaultdict(int)})
    live = set()
    nstep = 0
    for e, slots, new, rows in steps(a.dump):
        nstep += 1
        ids = e["req_ids"]
        # drop finished requests' buffers
        cur = set(ids)
        for gone in live - cur:
            for c in caps:
                for key in [k for k in req[c] if k[0] == gone]:
                    del req[c][key]
        live = cur
        t = e["t"]
        count = nstep > a.warmup
        bucket = "M<=4" if t <= 4 else ("M<=8" if t <= 8 else "M<=16" if t <= 16 else "M>16")
        b = tot[bucket]
        if count:
            b["steps"] += 1
        new_set = set(int(x) for x in new if x >= 0)
        for li, layer in enumerate(e["layers"]):
            union_all = set()
            for r in np.unique(rows):
                rid = ids[r] if 0 <= r < len(ids) else f"row{r}"
                sel = slots[li, rows == r]
                u = np.unique(sel[sel >= 0])
                keys = u.tolist()
                union_all.update(keys)
                for c in caps:
                    m = req[c][(rid, layer)].access(keys, new_set)
                    if count:
                        b["req"][c] += m
                        per_layer[layer]["req"][c] += m
                if count:
                    b["rows"] += len(keys)
                    per_layer[layer]["rows"] += len(keys)
            for c in caps:
                m = glob[c][layer].access(sorted(union_all), new_set)
                if count:
                    b["glob"][c] += m
        if count:
            b["new"] += len(new_set)
    out = {"steps": nstep, "buckets": {}, "per_layer": {}}
    for k, b in sorted(tot.items()):
        rows = max(b["rows"], 1)
        out["buckets"][k] = {
            "steps": b["steps"],
            "rows_per_step_layer": b["rows"] / max(b["steps"], 1) / len(e["layers"]),
            "miss_rate_req": {c: b["req"][c] / rows for c in caps},
            "miss_rate_global": {c: b["glob"][c] / rows for c in caps},
            "miss_rows_per_step_layer_req": {
                c: b["req"][c] / max(b["steps"], 1) / len(e["layers"]) for c in caps},
        }
    for layer, p in sorted(per_layer.items()):
        out["per_layer"][layer] = {c: p["req"][c] / max(p["rows"], 1) for c in caps}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
