#!/usr/bin/env python3
"""Phase 1 prototype: fused replica assignment for tiered MoE decode.

One kernel per routed layer replaces v1's standalone scheduler. It counts the
layer's routes, reduces the assignment to a min-max edge-orientation problem
over four ranks and six rank-pair classes, and solves it exactly by bounded
path reversal.

Unlike v1's greedy this needs no cost constants: hot experts are pinned to
their primary, every active cold expert costs the same Grace weight read, so
the objective is purely to minimise the maximum per-rank count of active cold
experts. See ``../README.md`` section 4.

The alignment fusion (folding both tiers' ``moe_align_block_size`` into this
same kernel) builds on the counts and maps produced here; it is the next step
and is not in this file yet.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

EP = 4
PAIRS = [(i, j) for i in range(EP) for j in range(i + 1, EP)]
PAIR_INDEX = {pair: index for index, pair in enumerate(PAIRS)}
NUM_PAIRS_PADDED = 8
MAX_REVERSALS = 64


def build_path_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate every reversal path, in the reference solver's search order.

    A path (u0 .. uL) moves one unit of load from u0 to uL by reversing one
    edge per hop. Because a simple path in K4 visits distinct ranks, each hop
    uses a distinct pair class, so a path's whole effect is one delta vector
    over the six classes: -1 where an edge leaves the lower rank, +1 where it
    leaves the higher one.

    Order is source, then path length, then ascending rank id, which is what
    ``solver_prototype.orient_path_reversal`` explores. The kernel then just
    takes the lowest usable index.
    """
    sources, targets, deltas = [], [], []
    for source in range(EP):
        others = [rank for rank in range(EP) if rank != source]
        chains = [(source, first) for first in others]
        chains += [
            (source, first, second)
            for first in others
            for second in others
            if second != first
        ]
        chains += [
            (source, first, second, third)
            for first in others
            for second in others
            for third in others
            if len({first, second, third}) == 3
        ]
        for chain in chains:
            delta = np.zeros(NUM_PAIRS_PADDED, dtype=np.int32)
            for hop in range(len(chain) - 1):
                head, tail = chain[hop], chain[hop + 1]
                index = PAIR_INDEX[(min(head, tail), max(head, tail))]
                delta[index] = -1 if head < tail else 1
            sources.append(source)
            targets.append(chain[-1])
            deltas.append(delta)
    return (
        np.asarray(sources, dtype=np.int32),
        np.asarray(targets, dtype=np.int32),
        np.stack(deltas),
    )


@triton.jit
def _assign_kernel(
    topk_ids_ptr,
    primary_rank_ptr,
    secondary_rank_ptr,
    primary_hot_ptr,
    path_source_ptr,
    path_target_ptr,
    path_delta_ptr,
    selected_rank_ptr,
    num_routes,
    NUM_EXPERTS: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
    PATH_BLOCK: tl.constexpr,
    NUM_PATHS: tl.constexpr,
    EP_SIZE: tl.constexpr,
    SCHEDULE: tl.constexpr,
    REVERSAL_LIMIT: tl.constexpr,
):
    experts = tl.arange(0, NUM_EXPERTS)
    routes = tl.arange(0, ROUTE_BLOCK)
    route_mask = routes < num_routes
    routed = tl.load(topk_ids_ptr + routes, mask=route_mask, other=-1).to(tl.int32)

    # Route histogram. One pass over the (expert, route) incidence matrix; at
    # decode shapes that is at most 256x128.
    hits = (experts[:, None] == routed[None, :]) & route_mask[None, :]
    counts = tl.sum(hits.to(tl.int32), 1)

    primary = tl.load(primary_rank_ptr + experts).to(tl.int32)
    secondary = tl.load(secondary_rank_ptr + experts).to(tl.int32)
    primary_hot = tl.load(primary_hot_ptr + experts).to(tl.int32)

    active = counts > 0
    selected = tl.where(active, primary, -1)

    if SCHEDULE:
        # Hot experts are pinned to their primary; only active cold experts
        # with a replica are free to move, and each is one unit of load on one of two ranks.
        cold = active & (primary_hot == 0)
        flexible = cold & (secondary >= 0)
        fixed = cold & (secondary < 0)
        low = tl.minimum(primary, secondary)
        high = tl.maximum(primary, secondary)

        pairs = tl.arange(0, PAIR_BLOCK)
        ranks = tl.arange(0, EP_SIZE)
        offsets = tl.sum(
            (fixed[None, :] & (primary[None, :] == ranks[:, None])).to(tl.int32), 1
        )

        # Per pair class: total edges, and how many currently point at the lower
        # rank. The initial orientation is "everything on its primary".
        totals = tl.zeros((PAIR_BLOCK,), dtype=tl.int32)
        split = tl.zeros((PAIR_BLOCK,), dtype=tl.int32)
        for lo in range(4):
          for hi in range(lo + 1, 4):
            index = lo * 4 - lo * (lo + 1) // 2 + (hi - lo - 1)
            member = flexible & (low == lo) & (high == hi)
            total = tl.sum(member.to(tl.int32), 0)
            at_low = tl.sum((member & (primary == lo)).to(tl.int32), 0)
            totals = tl.where(pairs == index, total, totals)
            split = tl.where(pairs == index, at_low, split)

        # Path reversal. Each iteration moves one unit of load from the lowest-id
        # maximum-loaded rank to a rank at least two lighter, along a path of at
        # most three classes. This is exact for min-max in-degree orientation.
        for _ in tl.range(0, REVERSAL_LIMIT):
            load = offsets
            for lo in range(4):
              for hi in range(lo + 1, 4):
                index = lo * 4 - lo * (lo + 1) // 2 + (hi - lo - 1)
                at_low = tl.sum(tl.where(pairs == index, split, 0), 0)
                total = tl.sum(tl.where(pairs == index, totals, 0), 0)
                load = load + tl.where(ranks == lo, at_low, 0)
                load = load + tl.where(ranks == hi, total - at_low, 0)

            peak = tl.max(load, 0)
            # Lowest-id rank attaining the maximum.
            source_id = tl.min(tl.where(load == peak, ranks, EP_SIZE), 0)

            # Candidate paths come from a host-built table ordered exactly as the
            # reference enumerates them: by source, then by length, then by
            # ascending rank id. Picking the lowest usable index therefore picks
            # the reference's path. A simple path in K4 visits distinct ranks, so
            # each hop uses a distinct pair class and the deltas never interact.
            paths = tl.arange(0, PATH_BLOCK)
            path_mask = paths < NUM_PATHS
            path_source = tl.load(path_source_ptr + paths, mask=path_mask, other=-1)
            path_target = tl.load(path_target_ptr + paths, mask=path_mask, other=-1)
            delta = tl.load(
                path_delta_ptr + paths[:, None] * PAIR_BLOCK + pairs[None, :],
                mask=path_mask[:, None],
                other=0,
            ).to(tl.int32)

            # A hop off the lower rank consumes an edge currently at the lower
            # rank, and vice versa.
            blocked = tl.sum(
                (
                    ((delta == -1) & (split[None, :] <= 0))
                    | ((delta == 1) & (totals[None, :] - split[None, :] <= 0))
                ).to(tl.int32),
                1,
            )
            target_load = tl.sum(
                tl.where(ranks[None, :] == path_target[:, None], load[None, :], 0), 1
            )
            usable = (
                path_mask
                & (path_source == source_id)
                & (target_load <= peak - 2)
                & (blocked == 0)
            )
            choice = tl.min(tl.where(usable, paths, NUM_PATHS), 0)
            found = tl.sum(tl.where(usable, 1, 0), 0)
            split = split + tl.sum(tl.where(paths[:, None] == choice, delta, 0), 0)

        # Realise the orientation. Within a pair class the edges are taken in
        # ascending expert id, so every rank derives the same assignment.
        for lo in range(4):
          for hi in range(lo + 1, 4):
            index = lo * 4 - lo * (lo + 1) // 2 + (hi - lo - 1)
            member = flexible & (low == lo) & (high == hi)
            at_low = tl.sum(tl.where(pairs == index, split, 0), 0)
            position = tl.cumsum(member.to(tl.int32), 0) - member.to(tl.int32)
            selected = tl.where(member, tl.where(position < at_low, lo, hi), selected)

    tl.store(selected_rank_ptr + experts, selected)


def fused_assign(
    topk_ids: torch.Tensor,
    primary_rank: torch.Tensor,
    secondary_rank: torch.Tensor,
    primary_hot: torch.Tensor,
    schedule: bool = True,
    reversal_limit: int = MAX_REVERSALS,
) -> torch.Tensor:
    """Return the chosen executing rank per expert, -1 where inactive."""
    num_experts = primary_rank.numel()
    flat = topk_ids.reshape(-1)
    table = _path_table(flat.device)
    selected = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    _assign_kernel[(1,)](
        flat,
        primary_rank,
        secondary_rank,
        primary_hot,
        table[0],
        table[1],
        table[2],
        selected,
        flat.numel(),
        NUM_EXPERTS=num_experts,
        ROUTE_BLOCK=triton.next_power_of_2(max(flat.numel(), 8)),
        PAIR_BLOCK=NUM_PAIRS_PADDED,
        PATH_BLOCK=triton.next_power_of_2(table[0].numel()),
        NUM_PATHS=table[0].numel(),
        EP_SIZE=EP,
        SCHEDULE=schedule,
        REVERSAL_LIMIT=reversal_limit,
        num_warps=4,
    )
    return selected


_PATH_TABLE_CACHE: dict = {}


def _path_table(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Device copy of the reversal path table; it is constant for a given EP."""
    key = str(device)
    if key not in _PATH_TABLE_CACHE:
        sources, targets, deltas = build_path_table()
        _PATH_TABLE_CACHE[key] = (
            torch.from_numpy(sources).to(device),
            torch.from_numpy(targets).to(device),
            torch.from_numpy(deltas).to(device).contiguous(),
        )
    return _PATH_TABLE_CACHE[key]


def reference_assign(
    counts: np.ndarray,
    primary_rank: np.ndarray,
    secondary_rank: np.ndarray,
    primary_hot: np.ndarray,
) -> np.ndarray:
    """Host reference: the same algorithm, written plainly."""
    from solver_prototype import orient_path_reversal

    num_experts = counts.size
    selected = np.where(counts > 0, primary_rank, -1).astype(np.int64)
    active = counts > 0
    cold = active & (primary_hot == 0)
    flexible = cold & (secondary_rank >= 0)
    fixed = cold & (secondary_rank < 0)

    offsets = np.zeros(EP, dtype=np.int64)
    for rank in range(EP):
        offsets[rank] = np.count_nonzero(fixed & (primary_rank == rank))

    low = np.minimum(primary_rank, secondary_rank)
    high = np.maximum(primary_rank, secondary_rank)
    totals = np.zeros(len(PAIRS), dtype=np.int64)
    at_low = np.zeros(len(PAIRS), dtype=np.int64)
    members = []
    for index, (lo, hi) in enumerate(PAIRS):
        member = flexible & (low == lo) & (high == hi)
        members.append(np.flatnonzero(member))
        totals[index] = member.sum()
        at_low[index] = (member & (primary_rank == lo)).sum()

    if totals.sum():
        split, _, _ = orient_path_reversal(offsets, totals, at_low)
        for index, (lo, hi) in enumerate(PAIRS):
            for position, expert in enumerate(members[index]):
                selected[expert] = lo if position < split[index] else hi
    del num_experts
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--concurrency", type=int, nargs="+", default=(1, 4))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from replay_exact import (
        NUM_LAYERS,
        load_placement,
        load_requests,
        route_counts,
        sample_steps,
    )

    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    heldout = load_requests(args.trace_dir, "heldout")
    owners, hot, secondary = load_placement(args.placement)

    report: dict = {"placement": str(args.placement), "workloads": {}}
    for concurrency in args.concurrency:
        steps = sample_steps(heldout, args.steps, concurrency, rng)
        counts = route_counts(steps)
        checked = 0
        mismatched = 0
        for step in range(steps.shape[0]):
            for layer in range(NUM_LAYERS):
                routes = steps[step, :, layer, :].reshape(-1).astype(np.int32)
                primary = owners[layer].astype(np.int32)
                replica = secondary[layer].astype(np.int32)
                is_hot = hot[layer].astype(np.int32)

                expected = reference_assign(
                    counts[step, layer].astype(np.int64),
                    primary.astype(np.int64),
                    replica.astype(np.int64),
                    is_hot.astype(np.int64),
                )
                actual = fused_assign(
                    torch.from_numpy(routes).to(device),
                    torch.from_numpy(primary).to(device),
                    torch.from_numpy(replica).to(device),
                    torch.from_numpy(is_hot).to(device),
                ).cpu().numpy()
                checked += 1
                if not np.array_equal(actual.astype(np.int64), expected):
                    mismatched += 1
                    if mismatched == 1:
                        bad = np.flatnonzero(actual.astype(np.int64) != expected)
                        print(
                            f"first mismatch at step {step} layer {layer}: "
                            f"{bad.size} experts differ, e.g. expert {bad[0]} "
                            f"kernel={actual[bad[0]]} reference={expected[bad[0]]}"
                        )
        name = f"c{concurrency}"
        report["workloads"][name] = {
            "layers_checked": checked,
            "mismatched_layers": mismatched,
        }
        print(f"{name}: {checked - mismatched}/{checked} layers match the reference")

    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
