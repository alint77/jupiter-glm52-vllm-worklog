"""Real per-expert token counts for prefill chunks, from the GLM-5.3 route
capture on the agentic task set (glm53-route-cap/merged: per request a
[tokens, 78 layers, top-8] uint8 array), mapped to GPUs with the served
placement profile. Chunks start at a random offset after the first 2048
tokens of a request (with prefix caching, a request's opening -- system
prompt, tool definitions -- is cached and not prefilled).
Writes counts_<chunk>.npy: [samples, 75 MoE layers, 4 ranks, 64 local
experts] (local order = ascending global id), for the grouped benchmark."""
import glob, json, sys
import numpy as np

CAP = "/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged-noloop"
prof = json.load(open("agent_space/profiles/glm53-w4a16-agentic-3239-r2000.json"))
layers = [int(l) for l in prof["routed_layers"]]
owners = np.array(prof["owners"])            # [75, 256]
hot = [set(h) for h in prof["hot_experts"]]  # per layer, all GPUs
local = np.zeros_like(owners)                # global id -> local slot on its owner
for li in range(len(layers)):
    for r in range(4):
        ids = np.flatnonzero(owners[li] == r)
        local[li, ids] = np.arange(len(ids))
files = sorted(glob.glob(f"{CAP}/*.npy"))
out = sys.argv[1] if len(sys.argv) > 1 else "/e/fscratch/profound/naeimitabiei1/ingraph-prefetch/counts"
for chunk in (512, 1024, 2048, 4096):
    samples = []
    rng = np.random.default_rng(chunk)
    for f in files:
        a = np.load(f, mmap_mode="r")
        if a.shape[0] < 2048 + chunk:
            continue
        o = int(rng.integers(2048, a.shape[0] - chunk + 1))
        c = np.zeros((len(layers), 4, 64), np.int32)
        for li, layer in enumerate(layers):
            ids = np.asarray(a[o:o + chunk, layer]).reshape(-1).astype(np.int64)
            np.add.at(c[li], (owners[li, ids], local[li, ids]), 1)
        samples.append(c)
    s = np.stack(samples)
    np.save(f"{out}_{chunk}.npy", s)
    flat = s.reshape(-1)
    pct = np.percentile(flat, [10, 50, 90, 99, 100])
    per_rank_max = s.max(axis=3)  # [samples, layers, ranks]
    buckets = [(0, 0), (1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10**9)]
    hist = " ".join(f"{lo}-{hi if hi < 10**9 else '':}:{((flat >= lo) & (flat <= hi)).mean():.0%}"
                    for lo, hi in buckets)
    print(f"chunk {chunk}: {len(samples)} requests; tokens/expert mean {flat.mean():.1f}, "
          f"p10/p50/p90/p99/max {pct.astype(int).tolist()}; busiest expert per GPU "
          f"median {np.median(per_rank_max):.0f}")
    print(f"   share of experts by tokens: {hist}")
    # per-GPU imbalance in routed tokens (compute) per layer
    tot = s.sum(axis=3)
    print(f"   routed tokens per GPU per layer: mean {tot.mean():.0f}, busiest/mean "
          f"{(tot.max(axis=2) / tot.mean(axis=2)).mean():.2f}")
