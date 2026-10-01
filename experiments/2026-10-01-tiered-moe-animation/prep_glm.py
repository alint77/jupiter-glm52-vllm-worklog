#!/usr/bin/env python3
"""GLM-5.3 routing skew for the frequency slide, from the routing capture."""

import json
from pathlib import Path

import numpy as np

TRACE = Path("/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged")
manifest = json.loads((TRACE / "manifest.json").read_text())
records = manifest["records"] if isinstance(manifest, dict) else manifest
first = np.load(TRACE / records[0]["file"])
print("trace shape", first.shape, first.dtype, "max id", first.max())
n_layers, E = first.shape[1], int(first.max()) + 1
E = 256 if E <= 256 else E
counts = np.zeros((n_layers, E), np.int64)
tokens = 0
for r in records:
    a = np.load(TRACE / r["file"])
    tokens += a.shape[0]
    for l in range(n_layers):
        ids = a[:, l, :].ravel()
        ids = ids[ids >= 0]
        counts[l] += np.bincount(ids, minlength=E)[:E]
# Dense layers have no routing; uint8 traces record them as expert 0.
routed = [l for l in range(n_layers) if counts[l].sum() > 0 and counts[l].max() < 0.5 * counts[l].sum()]
print("routed layers", len(routed), "first", routed[:3])
share = counts[routed] / counts[routed].sum(1, keepdims=True)
srt = -np.sort(-share, axis=1)
mid = len(routed) // 2
out = {
    "layer_shown": int(routed[mid]),
    "num_experts": E,
    "sorted_share_layer": srt[mid].round(6).tolist(),
    "cold_share_least_used_half": round(float(srt[:, E // 2:].sum(1).mean()), 3),
    "tokens": tokens,
}
Path("glm.json").write_text(json.dumps(out))
print({k: v for k, v in out.items() if k != "sorted_share_layer"})
