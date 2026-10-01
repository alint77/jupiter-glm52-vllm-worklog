#!/usr/bin/env python3
"""One real 8-token step for the per-GPU layout scenes: which experts each GPU
owns, which are hot/cold/replicas, and which fire. Writes step.json."""

import json
from pathlib import Path

import numpy as np

TRACE = Path("/e/fscratch/profound/naeimitabiei1/mimo26-route-cap/merged")
PROFILE = Path(__file__).resolve().parents[2] / "profiles/mimo26-profile-3827-r1500.json"
STEP = 8

prof = json.loads(PROFILE.read_text())
layers = prof["routed_layers"]
owners = np.array(prof["owners"])
sec = np.array(prof["secondary_ranks"])
manifest = json.loads((TRACE / "manifest.json").read_text())
records = manifest["records"] if isinstance(manifest, dict) else manifest


def balance(cold, li):
    load = np.zeros(4, int)
    where = {}
    for e in sorted(cold, key=lambda e: sec[li, e] >= 0):
        o, r = owners[li, e], sec[li, e]
        dst = r if r >= 0 and load[r] < load[o] else o
        load[dst] += 1
        where[int(e)] = int(dst)
    return load, where


best = None
for rec in records:
    a = np.load(TRACE / rec["file"])
    for s in range(0, (a.shape[0] // STEP) * STEP, STEP):
        for li, layer in enumerate(layers):
            hot = set(prof["hot_experts"][li])
            ids = np.unique(a[s : s + STEP, layer, :])
            if not 47 <= len(ids) <= 51 or not 10 <= layer <= 55:
                continue
            cold = [int(e) for e in ids if e not in hot]
            per = np.bincount(owners[li, cold], minlength=4)
            load, where = balance(cold, li)
            gain = per.max() - load.max()
            if 7 <= len(cold) <= 12 and per.max() >= 4 and gain >= 2 and (best is None or gain > best[0]):
                best = (gain, li, layer, a[s : s + STEP, layer, :], ids, cold, per, load, where)
    if best and best[0] >= 3:
        break

gain, li, layer, toks, ids, cold, per, load, where = best
hot = set(prof["hot_experts"][li])
ranks = []
for r in range(4):
    owned = [int(e) for e in np.where(owners[li] == r)[0]]
    ranks.append({
        "owned": owned,
        "hot": [e for e in owned if e in hot],
        "cold": [e for e in owned if e not in hot],
        "replicas": [int(e) for e in np.where(sec[li] == r)[0]],
    })
out = {
    "layer": layer,
    "token0": sorted(int(e) for e in toks[0]),
    "active": sorted(int(e) for e in ids),
    "cold_active": cold,
    "owner": {str(e): int(owners[li, e]) for e in ids},
    "assigned": {str(k): v for k, v in where.items()},
    "no_rep": per.tolist(),
    "with_rep": load.tolist(),
    "ranks": ranks,
}
Path("step.json").write_text(json.dumps(out))
print(layer, len(ids), "cold", len(cold), per.tolist(), "->", load.tolist(),
      [ (len(r["owned"]), len(r["hot"]), len(r["cold"]), len(r["replicas"])) for r in ranks])
