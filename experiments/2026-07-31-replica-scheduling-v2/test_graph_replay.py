"""Does a captured graph produce correct results when routes change?

The served path captures once and replays with new topk_ids every step. The
correctness harness so far launches eagerly, so it would not catch state that
persists across replays -- for example scratch that is read before it is
written, or a shape baked in at capture.
"""
import numpy as np
import torch
import fused_assign_align as F

dev = torch.device("cuda")
rng = np.random.default_rng(11)
E, TOPK, BM, NT = 256, 8, 16, 16
primary = np.repeat(np.arange(4, dtype=np.int32), E // 4)
secondary = np.full(E, -1, dtype=np.int32)
pick = rng.random(E) < 0.4
secondary[pick] = (primary[pick] + 1 + rng.integers(0, 3, pick.sum())) % 4
secondary[secondary == primary] = -1
is_hot = (rng.random(E) < 0.5).astype(np.int32)
ep = 2
hot_map, cold_map = F.build_tier_maps(primary, is_hot.astype(bool), secondary, ep)
maps = F.TierMaps(torch.from_numpy(hot_map).to(dev), torch.from_numpy(cold_map).to(dev))
pt, st, ht = (torch.from_numpy(a).to(dev) for a in (primary, secondary, is_hot))

topk = torch.zeros((NT, TOPK), dtype=torch.int32, device=dev)
out = F.fused_assign_align(topk, pt, st, ht, maps, ep, BM, BM, num_warps=8)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    F.fused_assign_align(topk, pt, st, ht, maps, ep, BM, BM, num_warps=8, out=out)

bad = 0
for trial in range(200):
    routes = rng.integers(0, E, size=(NT, TOPK)).astype(np.int32)
    topk.copy_(torch.from_numpy(routes).to(dev))
    g.replay(); torch.cuda.synchronize()
    counts = np.bincount(routes.reshape(-1), minlength=E)
    exp = F.reference_assign(counts.astype(np.int64), primary.astype(np.int64),
                             secondary.astype(np.int64), is_hot.astype(np.int64))
    got = out.selected_rank.cpu().numpy().astype(np.int64)
    if not np.array_equal(got, exp):
        bad += 1
        if bad == 1:
            d = np.flatnonzero(got != exp)
            print(f"trial {trial}: {d.size} experts differ, first {d[0]}: got {got[d[0]]} want {exp[d[0]]}")
    # Every active expert must be executed exactly once across the four ranks.
    active = counts > 0
    total = np.zeros(E, dtype=np.int64)
    for r in range(4):
        total += (active & (is_hot != 0) & (primary == r)).astype(np.int64)
        total += (active & (is_hot == 0) & (exp == r)).astype(np.int64)
    assert np.array_equal(total, active.astype(np.int64)), "exactly-once violated"
if bad:
    raise SystemExit(f"graph replay: {bad}/200 trials diverged from the reference")
print("graph replay: 200/200 trials match the reference, exactly-once held")
