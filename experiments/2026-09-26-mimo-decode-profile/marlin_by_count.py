#!/usr/bin/env python3
"""Marlin time and efficiency as a function of how many experts a rank runs.

Joins two recordings of the same decode requests: torch-profiler traces (the
duration of every hot and cold Marlin launch, per rank and layer) and the
routed-expert trace (which experts each 8-token verify step activated). For
every (step, layer, rank) it reconstructs what that rank actually executed:

  hot   active experts whose primary owner is this rank and that are HBM
        resident (the profile's hot list, demoted to the runtime's slot count
        with the planner's own `_demote_overfilled_residency`);
  cold  active cold experts the replica assignment gave this rank, computed by
        running the production `tiered_moe_assign_align` op on the same
        inputs, so replicas are attributed exactly as the server did.

Routes and trace steps are aligned by the offset that maximises the
correlation between per-rank cold counts and cold Marlin time; a wrong offset
gives ~0 correlation, so the choice is unambiguous and is reported.

Bytes per expert are the resident mxfp4 layout: w13 13,369,344 B (weights +
e8m0 scales), w2 6,684,672 B. Every active expert fits one 16-row Marlin block
at 8 tokens, so each is read exactly once per launch. Roofs: HBM 3.626 TB/s,
C2C 0.409 TB/s (NUMA-bound probe, 2026-08-31).

    marlin_by_count.py --traces DIR --routes DIR --profile P.json --label L --out figs/
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from analyze import RANK_RE, load, steps

W13_BYTES = 13_369_344
W2_BYTES = 6_684_672
ROOF = {"hot": 3.626e12, "cold": 0.409e12}
EP = 4
STEP = 8
HOT_SLOTS = 3764


def runtime_placement(profile: dict):
    from vllm.model_executor.model_loader.tiered_moe_planner import (
        _demote_overfilled_residency,
    )

    owners = np.asarray(profile["owners"])
    secondary = np.asarray(profile["secondary_ranks"])
    layers = tuple(range(len(profile["routed_layers"])))
    hot = np.zeros(owners.shape, dtype=bool)
    for rank in range(EP):
        hot_map = {l: tuple(sorted(e for e in profile["hot_experts"][l] if owners[l, e] == rank))
                   for l in layers}
        excess = sum(len(v) for v in hot_map.values()) - HOT_SLOTS
        if excess > 0:
            hot_map = _demote_overfilled_residency(hot_map, layers, excess)
        for l, ids in hot_map.items():
            hot[l, list(ids)] = True
    return owners, secondary, hot


def tier_maps(owners, secondary, hot, layer, rank):
    n = owners.shape[1]
    hot_map = np.full(n, -1, dtype=np.int32)
    cold_map = np.full(n, -1, dtype=np.int32)
    hot_ids = [e for e in range(n) if owners[layer, e] == rank and hot[layer, e]]
    cold_ids = [e for e in range(n) if owners[layer, e] == rank and not hot[layer, e]]
    cold_ids += [e for e in range(n) if secondary[layer, e] == rank]
    hot_map[hot_ids] = np.arange(len(hot_ids))
    cold_map[cold_ids] = np.arange(len(cold_ids))
    return hot_map, cold_map


def executed_counts(route_steps: np.ndarray, owners, secondary, hot) -> np.ndarray:
    """[steps, layers, rank, tier(0 hot, 1 cold)] experts executed."""
    from vllm.model_executor.model_loader.tiered_moe_scheduler import allocate_fused_routing

    device = torch.device("cuda")
    n_steps, _, n_layers, _ = route_steps.shape
    n_exp = owners.shape[1]
    out = np.zeros((n_steps, n_layers, EP, 2), dtype=np.int32)
    routing = allocate_fused_routing(STEP * 8, n_exp, 16, 16, device)
    for layer in range(n_layers):
        prim = torch.from_numpy(owners[layer].astype(np.int32)).to(device)
        sec = torch.from_numpy(secondary[layer].astype(np.int32)).to(device)
        phot = torch.from_numpy(hot[layer].astype(np.int32)).to(device)
        hmap, cmap = (torch.from_numpy(m).to(device) for m in tier_maps(owners, secondary, hot, layer, 0))
        for s in range(n_steps):
            ids = route_steps[s, :, layer, :]
            topk = torch.from_numpy(ids.astype(np.int32)).to(device)
            torch.ops.vllm.tiered_moe_assign_align(
                topk, prim, sec, phot, hmap, cmap, routing.scratch, routing.selected_rank,
                routing.hot_out_map, routing.cold_out_map, routing.hot_sorted,
                routing.hot_expert_ids, routing.hot_num_post, routing.cold_sorted,
                routing.cold_expert_ids, routing.cold_num_post, 0, 16, 16, True)
            selected = routing.selected_rank.cpu().numpy()
            active = np.unique(ids)
            for e in active:
                if hot[layer, e]:
                    out[s, layer, owners[layer, e], 0] += 1
                else:
                    out[s, layer, selected[e], 1] += 1
    return out


def trace_marlin(trace_dir: Path) -> dict[int, np.ndarray]:
    """rank -> [steps, layers, tier, 2 (w13, w2)] Marlin durations in us."""
    out = {}
    for path in sorted(trace_dir.glob("*.pt.trace.json.gz")):
        rank = int(RANK_RE.search(path.name).group(1))
        rows = steps(load(path))
        arr = np.full((len(rows), 69, 2, 2), np.nan)
        for i, row in enumerate(rows):
            ops = sorted(row["phases"]["target"], key=lambda e: e["t"])
            layer, seen = -1, collections.Counter()
            left_has_marlin = False
            for op in ops:
                if "cross_device_reduce" in op["name"]:
                    if left_has_marlin:
                        seen = collections.Counter()
                        left_has_marlin = False
                    continue
                if "marlin_moe" not in op["name"]:
                    continue
                if not left_has_marlin:
                    layer += 1
                    left_has_marlin = True
                grid = (op["args"].get("grid") or [0])[0]
                tier = 0 if grid == 264 else 1
                if layer < 69 and seen[tier] < 2:
                    arr[i, layer, tier, seen[tier]] = (op["end"] - op["t"]) * 1000
                seen[tier] += 1
        out[rank] = arr
    return out


def align(counts: np.ndarray, marlin: dict[int, np.ndarray]) -> tuple[int, float]:
    n_trace = min(v.shape[0] for v in marlin.values())
    measured = np.stack([marlin[r][:n_trace, :, 1].sum(-1) for r in range(EP)], -1)
    best = (None, -1.0)
    for offset in range(0, counts.shape[0] - n_trace + 1):
        predicted = counts[offset:offset + n_trace, :, :, 1]
        mask = ~np.isnan(measured)
        corr = np.corrcoef(predicted[mask], measured[mask])[0, 1]
        if corr > best[1]:
            best = (offset, float(corr))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True, help="one request's route .npy")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    owners, secondary, hot = runtime_placement(profile)
    routes = np.load(args.routes)[:, profile["routed_layers"], :]
    usable = routes.shape[0] // STEP * STEP
    route_steps = routes[:usable].reshape(-1, STEP, len(profile["routed_layers"]), routes.shape[-1])
    marlin = trace_marlin(args.traces)
    n_trace = min(v.shape[0] for v in marlin.values())
    # Only the region that can hold the window needs the assignment replayed,
    # but the offset is unknown, so replay every step once.
    counts = executed_counts(route_steps, owners, secondary, hot)
    offset, corr = align(counts, marlin)
    print(f"{args.label}: {route_steps.shape[0]} route steps, {n_trace} traced steps; "
          f"offset {offset}, cold count vs time corr {corr:.3f}")
    samples = {"hot": collections.defaultdict(list), "cold": collections.defaultdict(list)}
    for rank in range(EP):
        c = counts[offset:offset + n_trace, :, rank, :]
        m = marlin[rank][:n_trace]
        for tier_index, tier in enumerate(("hot", "cold")):
            for s in range(n_trace):
                for layer in range(69):
                    w13, w2 = m[s, layer, tier_index]
                    if np.isnan(w13) or np.isnan(w2):
                        continue
                    samples[tier][int(c[s, layer, tier_index])].append((w13, w2))
    result = {"label": args.label, "offset": offset, "correlation": corr, "tiers": {}}
    for tier, by_n in samples.items():
        rows = {}
        for n in sorted(by_n):
            w13 = [x[0] for x in by_n[n]]
            w2 = [x[1] for x in by_n[n]]
            total = [a + b for a, b in by_n[n]]
            med = statistics.median(total)
            row = {"samples": len(total), "w13_us": statistics.median(w13),
                   "w2_us": statistics.median(w2), "total_us": med,
                   "total_p10": float(np.percentile(total, 10)),
                   "total_p90": float(np.percentile(total, 90))}
            if n:
                nbytes = n * (W13_BYTES + W2_BYTES)
                row |= {"gbps": nbytes / (med * 1e-6) / 1e9,
                        "roof_fraction": nbytes / (med * 1e-6) / ROOF[tier],
                        "sol_us": nbytes / ROOF[tier] * 1e6}
            rows[n] = row
        result["tiers"][tier] = rows
        xs = np.array([n for n in by_n for _ in by_n[n]])
        ys = np.array([a + b for n in by_n for a, b in by_n[n]])
        slope, intercept = np.polyfit(xs, ys, 1)
        result["tiers"][tier + "_fit"] = {"fixed_us": float(intercept), "per_expert_us": float(slope),
                                          "marginal_gbps": (W13_BYTES + W2_BYTES) / (slope * 1e-6) / 1e9}
        print(f"\n{tier}: time(w13+w2) = {intercept:.1f} us + {slope:.1f} us x experts "
              f"(marginal {result['tiers'][tier + '_fit']['marginal_gbps']:.0f} GB/s)")
        print(f"{'n':>3s} {'samples':>8s} {'w13 us':>7s} {'w2 us':>7s} {'total us':>9s} "
              f"{'p10-p90':>13s} {'GB/s':>7s} {'% roof':>7s}")
        for n, row in rows.items():
            extra = f"{row['gbps']:7.0f} {row['roof_fraction'] * 100:6.0f}%" if n else ""
            print(f"{n:3d} {row['samples']:8d} {row['w13_us']:7.1f} {row['w2_us']:7.1f} "
                  f"{row['total_us']:9.1f} {row['total_p10']:6.1f}-{row['total_p90']:<6.1f} {extra}")
    args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
