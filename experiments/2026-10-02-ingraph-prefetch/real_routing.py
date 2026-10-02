"""Real GLM-5.3 routing for prefill-chunk benchmarks.

Samples chunks from the agentic route capture (glm53-route-cap/merged-noloop,
requests that looped during capture removed: per
request [tokens, 78 layers, top-8] expert ids) at random offsets past each
request's first 2048 tokens (the opening is prefix-cached in serving), for a
random MoE layer and GPU; experts map to GPUs with the served profile.
"""
import glob
import json
from pathlib import Path

import numpy as np

CAP = "/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged-noloop"
PROFILE = (Path(__file__).resolve().parents[2] / "profiles" /
           "glm53-w4a16-agentic-3239-r2000.json")


class Sample:
    """One chunk of one layer on one GPU."""

    def __init__(self, layer, rank, topk_ids, local_map):
        self.layer, self.rank = layer, rank
        self.topk_ids = topk_ids    # [chunk, 8] int32 global expert ids
        self.local_map = local_map  # [256] int32: global -> local slot or -1

    def counts(self):
        """Tokens routed to each of this GPU's 64 experts."""
        loc = self.local_map[self.topk_ids.reshape(-1)]
        return np.bincount(loc[loc >= 0], minlength=64)


def samples(chunk, n, seed=0):
    prof = json.load(open(PROFILE))
    layers = [int(l) for l in prof["routed_layers"]]
    owners = np.array(prof["owners"])
    rng = np.random.default_rng(seed)
    files = [f for f in sorted(glob.glob(f"{CAP}/*.npy"))
             if np.load(f, mmap_mode="r").shape[0] >= 2048 + chunk]
    out = []
    for _ in range(n):
        a = np.load(files[rng.integers(len(files))], mmap_mode="r")
        o = int(rng.integers(2048, a.shape[0] - chunk + 1))
        li = int(rng.integers(len(layers)))
        rank = int(rng.integers(4))
        ids = np.asarray(a[o:o + chunk, layers[li]]).astype(np.int32)
        mine = np.flatnonzero(owners[li] == rank)
        local_map = np.full(256, -1, np.int32)
        local_map[mine] = np.arange(len(mine), dtype=np.int32)
        out.append(Sample(layers[li], rank, ids, local_map))
    return out
