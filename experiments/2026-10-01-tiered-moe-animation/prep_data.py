#!/usr/bin/env python3
"""Numbers the animation shows, from real MiMo-V2.6-Pro routing traces and the
production placement profile (mimo26-profile-3827-r1500.json)."""

import json
from pathlib import Path

import numpy as np

TRACE = Path("/e/fscratch/profound/naeimitabiei1/mimo26-route-cap/merged")
PROFILE = Path(__file__).resolve().parents[2] / "profiles/mimo26-profile-3827-r1500.json"
STEP = 8
E = 384
rng = np.random.default_rng(0)

prof = json.loads(PROFILE.read_text())
layers = prof["routed_layers"]
owners = np.array(prof["owners"])  # [layer_idx, expert] -> rank
hot = [set(h) for h in prof["hot_experts"]]
sec = np.array(prof["secondary_ranks"])  # -1 if no replica

manifest = json.loads((TRACE / "manifest.json").read_text())
records = manifest["records"] if isinstance(manifest, dict) else manifest
routes = [np.load(TRACE / r["file"]) for r in records]

counts = np.zeros((len(layers), E), np.int64)
for a in routes:
    for li, layer in enumerate(layers):
        counts[li] += np.bincount(a[:, layer, :].ravel(), minlength=E)

share = counts / counts.sum(1, keepdims=True)
sorted_share = -np.sort(-share, axis=1)
hot_mask = np.zeros((len(layers), E), bool)
for li, h in enumerate(hot):
    hot_mask[li, list(h)] = True
cold_share_profile = float((share * ~hot_mask).sum(1).mean())
least_half = float(sorted_share[:, E // 2 :].sum(1).mean())
hot_frac = float(hot_mask.mean())

# Per verify step (8 consecutive tokens): unique active experts per rank.
no_rep, with_rep, act1, act8, cold_total = [], [], [], [], []
example = None
for a in routes:
    n = (a.shape[0] // STEP) * STEP
    for s in range(0, n, STEP):
        for li, layer in enumerate(layers):
            ids = np.unique(a[s : s + STEP, layer, :])
            act8.append(len(ids))
            cold = [e for e in ids if not hot_mask[li, e]]
            cold_total.append(len(cold))
            per = np.bincount(owners[li, cold], minlength=4) if cold else np.zeros(4, int)
            no_rep.append(per.max())
            load = np.zeros(4, int)
            for e in sorted(cold, key=lambda e: sec[li, e] >= 0):
                o, r = owners[li, e], sec[li, e]
                load[r if r >= 0 and load[r] < load[o] else o] += 1
            with_rep.append(load.max())
            if example is None and per.max() >= 4 and load.max() <= per.max() - 2:
                example = {"layer": layer, "no_rep": per.tolist(), "with_rep": load.tolist()}
        act1.extend(len(np.unique(a[s, l, :])) for l in layers[:1])

no_rep, with_rep, cold_total = map(np.array, (no_rep, with_rep, cold_total))
out = {
    "layer_shown": layers[len(layers) // 2],
    "sorted_share_layer": sorted_share[len(layers) // 2].round(6).tolist(),
    "hot_fraction_of_experts": round(hot_frac, 3),
    "cold_share_profile": round(cold_share_profile, 3),
    "cold_share_least_used_half": round(least_half, 3),
    "active_per_layer_1tok": 8,
    "active_per_layer_8tok_mean": round(float(np.mean(act8)), 1),
    "cold_per_step_mean": round(float(cold_total.mean()), 2),
    "max_cold_rank_no_replicas_mean": round(float(no_rep.mean()), 2),
    "max_cold_rank_replicas_mean": round(float(with_rep.mean()), 2),
    "ideal_max_cold_rank_mean": round(float(np.ceil(cold_total / 4).mean()), 2),
    "example_step": example,
    "tokens": int(sum(a.shape[0] for a in routes)),
}
Path("data.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "sorted_share_layer"})
