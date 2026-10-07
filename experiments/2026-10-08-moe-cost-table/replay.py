"""Score the decode kernel's replica balancer offline on captured GLM routing.

For sampled (step, layer) cases of the live Claude Code capture, the kernel's
assignment (../2026-10-04-tiered-decode-kernel-review/profile_grid.assign: hot
experts on their owner, flexible cold experts moved by path reversal over a
cost table) is computed with each candidate table; every assignment is then
scored with the *measured* INT4 cost of each rank's (hot, cold) cell
(grid-<job>.jsonl). Reported per step: the sum over layers of the slowest rank's
cost (what the MoE all-reduce waits for) and of the ranks' mean, and the lower
bound with perfect balance of the flexible experts under the measured table.

    replay.py <grid jsonl> [--slots 3676] [--steps 4000]
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-04-tiered-decode-kernel-review"))
import profile_grid as pg  # noqa: E402


def table(grid, fmt):
    rows = [json.loads(l) for l in open(grid) if l.strip()]
    t = np.full((25, 7), np.nan)
    for h in range(25):
        for c in range(7):
            v = [r["us"] for r in rows if r["fmt"] == fmt and r["hot"] == h and r["cold"] == c]
            if v:
                t[h, c] = np.median(v)
    t[0, 0] = 0.0
    # non-decreasing in h and c, as the shipped table is made
    t = np.maximum.accumulate(np.maximum.accumulate(t, axis=0), axis=1)
    return t


def runtime_hot(profile, slots):
    """tiered_moe_planner: each rank's profile list, then extra slots promoted
    one per layer round-robin in owned (id) order, skipping full layers."""
    owners = np.asarray(profile["owners"])
    layers, n = owners.shape
    hot = np.zeros(owners.shape, dtype=bool)
    for r in range(4):
        lists = [[e for e in profile["hot_experts"][li] if owners[li, e] == r] for li in range(layers)]
        owned = [[e for e in range(n) if owners[li, e] == r] for li in range(layers)]
        extra = slots - sum(map(len, lists))
        assert extra >= 0
        while extra:
            for li in range(layers):
                if not extra:
                    break
                have = set(lists[li])
                nxt = next((e for e in owned[li] if e not in have), None)
                if nxt is not None:
                    lists[li].append(nxt)
                    extra -= 1
        for li in range(layers):
            hot[li, lists[li]] = True
    return hot


def expand(base):
    h, c = np.arange(65)[:, None], np.arange(65)[None, :]
    return (base[np.minimum(h, 24), np.minimum(c, 6)] + 8 * np.maximum(h - 24, 0)
            + 40 * np.maximum(c - 6, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--slots", type=int, default=3676)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    profile, primary, secondary, _, shipped, pair, delta, target = pg.configuration()
    hot = runtime_hot(profile, a.slots)
    int4 = table(a.grid, "int4")
    mx = table(a.grid, "mxfp4")
    src = (HERE.parents[2] / "vllm/model_executor/layers/fused_moe/tiered_decode/tiered_decode.cu").read_text()
    body = src.split("COST_US[COST_HOT][COST_COLD] = {")[1].split("};")[0]
    shipped_base = np.asarray([int(v) for v in re.findall(r"\d+", body)]).reshape(25, 7)
    print("shipped / measured MXFP4 / measured INT4 at a few cells (us):")
    for h, c in ((8, 0), (8, 2), (12, 2), (4, 4), (16, 1), (0, 3), (24, 0)):
        print(f"  ({h:2d},{c}) {shipped_base[h, c]:5.0f} {mx[h, c]:7.1f} {int4[h, c]:7.1f}")
    true = expand(int4)
    cand = {"shipped": shipped, "GLM INT4": expand(np.round(int4).astype(int))}
    info = json.loads((HERE.parent / "2026-10-04-tiered-decode-kernel-review/profile-grid.json").read_text())
    files = [f for f in info["files"] if "claude-glm53-df2-dcp4-cap" in f]
    rng = np.random.default_rng(0)
    res = {k: [0.0, 0.0] for k in cand}
    nsteps = 0
    per_file = max(1, a.steps // len(files))
    for f in files:
        arr = np.load(f, mmap_mode="r")
        steps = arr.shape[0] // 8
        for step in rng.choice(steps, size=min(per_file, steps), replace=False):
            nsteps += 1
            for li in range(75):
                ids = np.asarray(arr[step * 8:(step + 1) * 8, li + 3], dtype=np.int64)
                counts = np.bincount(ids[ids >= 0].ravel(), minlength=256)
                for name, cost in cand.items():
                    hs, cs, _ = pg.assign(counts, primary[li], secondary[li], hot[li], cost,
                                          pair, delta, target)
                    t = true[hs, cs]
                    res[name][0] += t.max()
                    res[name][1] += t.mean()
    print(f"\n{nsteps} steps x 75 layers, {a.slots} hot slots per rank, scored with measured INT4")
    for name, (mx_, mean) in res.items():
        print(f"  table {name:9s}: slowest rank {mx_ / nsteps / 1e3:.3f} ms/step, mean rank "
              f"{mean / nsteps / 1e3:.3f}, spread {(mx_ - mean) / nsteps / 1e3:.3f}")


main()
