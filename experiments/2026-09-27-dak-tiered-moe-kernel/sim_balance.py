#!/usr/bin/env python3
"""How much a cost-aware replica assignment could cut MoE imbalance, offline.

Replays held-out MiMo decode steps (8-token verify) through the deployed
placement (profile + 1500 replicas per GPU). Per step and layer, each GPU runs
its active hot experts (pinned) and some active cold experts; a cold expert
with a replica can run on either holder. Two assignments:

  count  today's rule: minimise the largest number of cold experts on a GPU
         (the runtime's path reversal, replayed exactly)
  time   minimise the largest predicted layer time max_r T(hot_r, cold_r),
         exhaustively over the flexible experts (greedy above 12 of them)
  reversal  what the route kernel does: path reversal on predicted time

T(h, c) is the one-kernel path's measured graph-replay time per layer
(tgrid.json), interpolated in h; the assignments balance the table the kernel
embeds, and every outcome is costed on the measured grid. The layer costs max over GPUs, since every
GPU waits for the slowest in the closing all-reduce.

    sim_balance.py --trace-dir /e/fscratch/.../mimo26-route-cap/merged \
        --profile agent_space/profiles/mimo26-profile-3827-r1500.json --steps 600
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "2026-09-26-mimo-routing-profile"))
from mimo_replicas import EP, active_mask, load_steps  # noqa: E402

from vllm.model_executor.model_loader.tiered_moe_scheduler import (  # noqa: E402
    MAX_REVERSALS,
    NUM_PAIRS_PADDED,
    PAIR_INDEX,
    build_path_table,
)

PATH_SOURCE, PATH_TARGET, PATH_DELTA = build_path_table()
LOW_INC = np.zeros((NUM_PAIRS_PADDED, EP), dtype=np.int64)
HIGH_INC = np.zeros((NUM_PAIRS_PADDED, EP), dtype=np.int64)
for (_lo, _hi), _index in PAIR_INDEX.items():
    LOW_INC[_index, _lo] = 1
    HIGH_INC[_index, _hi] = 1


def cost_table(path: Path, kernel: bool) -> np.ndarray:
    """T[h, c] in us for h, c < 65. kernel=True: exactly what tiered_decode.cu
    embeds (gen_cost_table.py's rounded, non-decreasing 25 x 7 table, +8 us per
    hot and +40 per cold beyond it); else the raw measured grid, interpolated."""
    rows = json.loads(path.read_text())
    grid = {(r["hot"], r["cold"]): r["us"] for r in rows}
    grid[(0, 0)] = 0.0
    hs = sorted({h for h, _ in grid})
    base = np.zeros((25, 7))
    for c in range(7):
        base[:, c] = np.interp(np.arange(25), hs, [grid[(h, c)] for h in hs])
    if kernel:
        base = np.rint(np.maximum.accumulate(np.maximum.accumulate(base, 0), 1))
    h = np.arange(65)[:, None]
    c = np.arange(65)[None, :]
    return (base[np.minimum(h, 24), np.minimum(c, 6)]
            + 8.0 * np.maximum(h - 24, 0) + 40.0 * np.maximum(c - 6, 0))


def count_assign(fixed_cold: np.ndarray, flex: list[tuple[int, int]]) -> np.ndarray:
    """The runtime's path reversal (_assign_kernel), replayed on counts."""
    totals = np.zeros(NUM_PAIRS_PADDED, dtype=np.int64)
    split = np.zeros(NUM_PAIRS_PADDED, dtype=np.int64)
    for a, b in flex:
        index = PAIR_INDEX[(min(a, b), max(a, b))]
        totals[index] += 1
        split[index] += a < b
    for _ in range(MAX_REVERSALS):
        load = fixed_cold + LOW_INC.T @ split + HIGH_INC.T @ (totals - split)
        peak = load.max()
        source = int(np.argmax(load))
        blocked = (((PATH_DELTA == -1) & (split[None] <= 0))
                   | ((PATH_DELTA == 1) & ((totals - split)[None] <= 0))).any(1)
        usable = (PATH_SOURCE == source) & (load[PATH_TARGET] <= peak - 2) & ~blocked
        if not usable.any():
            break
        split = split + PATH_DELTA[np.argmax(usable)]
    return fixed_cold + LOW_INC.T @ split + HIGH_INC.T @ (totals - split)


def time_reversal(hot: np.ndarray, fixed_cold: np.ndarray, flex, T) -> np.ndarray:
    """Path reversal driven by predicted time: move one cold expert off the
    slowest GPU along a path whose end stays faster than the source was."""
    totals = np.zeros(NUM_PAIRS_PADDED, dtype=np.int64)
    split = np.zeros(NUM_PAIRS_PADDED, dtype=np.int64)
    for a, b in flex:
        index = PAIR_INDEX[(min(a, b), max(a, b))]
        totals[index] += 1
        split[index] += a < b
    for _ in range(MAX_REVERSALS):
        cold = fixed_cold + LOW_INC.T @ split + HIGH_INC.T @ (totals - split)
        t = T[hot, cold]
        source = int(np.argmax(t))
        after = T[hot, cold + 1]
        blocked = (((PATH_DELTA == -1) & (split[None] <= 0))
                   | ((PATH_DELTA == 1) & ((totals - split)[None] <= 0))).any(1)
        usable = (PATH_SOURCE == source) & (after[PATH_TARGET] < t[source]) & ~blocked
        if not usable.any():
            break
        gain = np.where(usable, after[PATH_TARGET], np.inf)
        split = split + PATH_DELTA[np.argmin(gain)]
    return fixed_cold + LOW_INC.T @ split + HIGH_INC.T @ (totals - split)


def time_assign(hot: np.ndarray, fixed_cold: np.ndarray, flex, T) -> np.ndarray:
    if len(flex) <= 12:
        best, best_cold = None, None
        for choice in itertools.product((0, 1), repeat=len(flex)):
            cold = fixed_cold.copy()
            for (a, b), pick in zip(flex, choice):
                cold[(a, b)[pick]] += 1
            t = T[hot, cold]
            key = (t.max(), t.sum())
            if best is None or key < best:
                best, best_cold = key, cold
        return best_cold
    cold = fixed_cold.copy()   # greedy: each expert to the holder that ends lower
    for a, b in flex:
        ta = T[hot[a], cold[a] + 1]
        tb = T[hot[b], cold[b] + 1]
        cold[a if ta <= tb else b] += 1
    return cold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    profile = json.loads(args.profile.read_text())
    layers = [int(l) for l in profile["routed_layers"]]
    owners = np.asarray(profile["owners"])
    secondary = np.asarray(profile["secondary_ranks"])
    hot_mask = np.zeros(owners.shape, dtype=bool)
    for li, ids in enumerate(profile["hot_experts"]):
        hot_mask[li, ids] = True
    steps = load_steps(args.trace_dir, "heldout", layers)
    rng = np.random.default_rng(0)
    steps = steps[rng.choice(len(steps), min(args.steps, len(steps)), replace=False)]
    active = active_mask(steps, owners.shape[1])       # [steps, layers, experts]
    T = cost_table(HERE / "tgrid.json", kernel=True)        # what the kernel balances
    measured = cost_table(HERE / "tgrid.json", kernel=False)  # what the layer costs

    per_layer = {"count": [], "time": [], "reversal": [], "mean": []}
    for s in range(active.shape[0]):
        for li in range(len(layers)):
            act = active[s, li]
            hot = np.array([(act & hot_mask[li] & (owners[li] == r)).sum() for r in range(EP)])
            cold_act = act & ~hot_mask[li]
            fixed = np.array([(cold_act & (secondary[li] < 0) & (owners[li] == r)).sum() for r in range(EP)])
            flex = [(int(owners[li, e]), int(secondary[li, e]))
                    for e in np.flatnonzero(cold_act & (secondary[li] >= 0))]
            c1 = count_assign(fixed, flex)
            c2 = time_assign(hot, fixed, flex, T)
            c3 = time_reversal(hot, fixed, flex, T)
            t1, t2 = measured[hot, c1], measured[hot, c2]
            per_layer["reversal"].append(measured[hot, c3].max())
            per_layer["count"].append(t1.max())
            per_layer["time"].append(t2.max())
            per_layer["mean"].append(t1.mean())
    n = active.shape[0]
    res = {k: float(np.sum(v) / n / 1000) for k, v in per_layer.items()}   # ms per step
    print(f"{n} held-out steps x {len(layers)} layers, MoE ms per step (sum of per-layer max over GPUs):")
    print(f"  today's count-balanced assignment : {res['count']:.2f}")
    print(f"  time-balanced assignment          : {res['time']:.2f}  ({res['count'] - res['time']:+.2f} saved)")
    print(f"  time-driven path reversal         : {res['reversal']:.2f}  ({res['count'] - res['reversal']:+.2f} saved)")
    print(f"  mean over GPUs (no waiting at all): {res['mean']:.2f}  (imbalance today {res['count'] - res['mean']:.2f})")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
