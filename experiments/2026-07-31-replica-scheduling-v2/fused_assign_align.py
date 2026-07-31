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
from typing import NamedTuple

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
def _emit_alignment(
    routed,
    route_mask,
    within,
    counts,
    mine,
    local,
    num_routes,
    scratch_ptr,
    sorted_ptr,
    expert_ids_ptr,
    num_post_ptr,
    NUM_EXPERTS: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_SORTED: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """Write one tier's Marlin routing metadata.

    Reproduces ``moe_align_block_size(..., ignore_invalid_experts=True)``:
    ``sorted_ptr`` holds flattened route indices grouped by expert and padded to
    ``BLOCK_M`` with the sentinel ``num_routes``; ``expert_ids_ptr`` holds the
    tier-local expert index per block, -1 past the end.

    The counts come from the caller, which already built them for the
    assignment, so this costs no second scan of the route set.
    """
    experts = tl.arange(0, NUM_EXPERTS)
    routes = tl.arange(0, ROUTE_BLOCK)

    blocks = tl.where(mine, (counts + BLOCK_M - 1) // BLOCK_M, 0)
    # Blocks are laid out in ascending *global* expert id. The reference orders
    # them by tier-local index instead, which for the cold tier differs because
    # replica copies are appended after this rank's primary-cold experts.
    # Either layout is correct: Marlin reads expert_ids[b] for each block and
    # scatters its output by route index, so block order is not observable in
    # the result. Matching the reference here would cost a 256x256 reduction to
    # rank experts by local index, which measured as this kernel's single
    # largest cost. The correctness harness compares the per-expert partition
    # rather than the buffer bit-for-bit.
    block_start = tl.cumsum(blocks, 0) - blocks
    total_blocks = tl.sum(blocks, 0)

    slots = tl.arange(0, MAX_SORTED)
    tl.store(sorted_ptr + slots, tl.zeros((MAX_SORTED,), dtype=tl.int32) + num_routes)
    # The scatter below overwrites part of the padding just written, and the two
    # stores are spread over different threads.
    tl.debug_barrier()

    # Per-route slot base and tier membership, by gather rather than by an
    # [routes, experts] incidence matrix - that matrix was built once per tier
    # and measured as the second largest cost in this kernel.
    tl.store(
        scratch_ptr + NUM_EXPERTS + experts,
        tl.where(mine, block_start * BLOCK_M, -1),
    )
    tl.debug_barrier()
    base = tl.load(
        scratch_ptr + NUM_EXPERTS + routed, mask=route_mask, other=-1
    ).to(tl.int32)
    keep = route_mask & (base >= 0)
    tl.store(sorted_ptr + tl.where(keep, base + within, 0), routes, mask=keep)

    # Per-block expert index by scatter-difference plus prefix sum, rather than
    # a [blocks, experts] range search: add local+1 at an expert's first block
    # and subtract it one past its last, so the running sum carries local+1
    # across exactly that expert's blocks and 0 elsewhere.
    span = tl.arange(0, MAX_BLOCKS)
    tl.store(
        scratch_ptr + 2 * NUM_EXPERTS + span,
        tl.zeros((MAX_BLOCKS,), dtype=tl.int32),
    )
    tl.debug_barrier()
    tl.atomic_add(
        scratch_ptr + 2 * NUM_EXPERTS + block_start, local + 1, mask=mine
    )
    tl.atomic_add(
        scratch_ptr + 2 * NUM_EXPERTS + block_start + blocks,
        -(local + 1),
        mask=mine & (block_start + blocks < MAX_BLOCKS),
    )
    tl.debug_barrier()
    marks = tl.load(scratch_ptr + 2 * NUM_EXPERTS + span).to(tl.int32)
    tl.store(expert_ids_ptr + span, tl.cumsum(marks, 0) - 1)
    tl.store(num_post_ptr, total_blocks * BLOCK_M)


@triton.jit
def _assign_kernel(
    topk_ids_ptr,
    primary_rank_ptr,
    secondary_rank_ptr,
    primary_hot_ptr,
    path_source_ptr,
    path_target_ptr,
    path_delta_ptr,
    hot_map_ptr,
    cold_map_ptr,
    scratch_ptr,
    selected_rank_ptr,
    hot_sorted_ptr,
    hot_expert_ids_ptr,
    hot_num_post_ptr,
    cold_sorted_ptr,
    cold_expert_ids_ptr,
    cold_num_post_ptr,
    num_routes,
    NUM_EXPERTS: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
    PATH_BLOCK: tl.constexpr,
    NUM_PATHS: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EP_RANK: tl.constexpr,
    SCHEDULE: tl.constexpr,
    REVERSAL_LIMIT: tl.constexpr,
    BLOCK_M_HOT: tl.constexpr,
    MAX_SORTED_HOT: tl.constexpr,
    MAX_BLOCKS_HOT: tl.constexpr,
    BLOCK_M_COLD: tl.constexpr,
    MAX_SORTED_COLD: tl.constexpr,
    MAX_BLOCKS_COLD: tl.constexpr,
):
    experts = tl.arange(0, NUM_EXPERTS)
    routes = tl.arange(0, ROUTE_BLOCK)
    route_mask = routes < num_routes
    routed = tl.load(topk_ids_ptr + routes, mask=route_mask, other=-1).to(tl.int32)

    # Route histogram by atomic scatter rather than a [experts, routes]
    # incidence matrix: 128 atomics against 32k compares at c4, and still
    # deterministic because addition is order independent.
    tl.store(scratch_ptr + experts, tl.zeros((NUM_EXPERTS,), dtype=tl.int32))
    tl.debug_barrier()
    tl.atomic_add(scratch_ptr + routed, 1, mask=route_mask)
    tl.debug_barrier()
    counts = tl.load(scratch_ptr + experts).to(tl.int32)

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
        #
        # The load vector is rebuilt from two reductions against a precomputed
        # pair/rank incidence rather than twelve scalar reductions per class,
        # because block-wide reductions dominate this kernel's cost.
        low_hot = tl.zeros((PAIR_BLOCK, EP_SIZE), dtype=tl.int32)
        high_hot = tl.zeros((PAIR_BLOCK, EP_SIZE), dtype=tl.int32)
        for lo in range(4):
          for hi in range(lo + 1, 4):
            index = lo * 4 - lo * (lo + 1) // 2 + (hi - lo - 1)
            here = pairs[:, None] == index
            low_hot = tl.where(here & (ranks[None, :] == lo), 1, low_hot)
            high_hot = tl.where(here & (ranks[None, :] == hi), 1, high_hot)

        paths = tl.arange(0, PATH_BLOCK)
        path_mask = paths < NUM_PATHS
        path_source = tl.load(path_source_ptr + paths, mask=path_mask, other=-1)
        path_target = tl.load(path_target_ptr + paths, mask=path_mask, other=-1)
        delta = tl.load(
            path_delta_ptr + paths[:, None] * PAIR_BLOCK + pairs[None, :],
            mask=path_mask[:, None],
            other=0,
        ).to(tl.int32)

        # Data-dependent trip count. This is intra-kernel control flow, so CUDA
        # graph capture is unaffected, and it saves the iterations a fixed trip
        # count would waste: convergence takes 3 reversals on average against a
        # measured worst case of 17.
        step = tl.sum(tl.zeros((PAIR_BLOCK,), dtype=tl.int32), 0)
        improving = step + 1
        while (step < REVERSAL_LIMIT) & (improving > 0):
            load = (
                offsets
                + tl.sum(split[:, None] * low_hot, 0)
                + tl.sum((totals - split)[:, None] * high_hot, 0)
            )
            peak = tl.max(load, 0)
            # Lowest-id rank attaining the maximum.
            source_id = tl.min(tl.where(load == peak, ranks, EP_SIZE), 0)

            # Candidate paths come from a host-built table ordered exactly as the
            # reference enumerates them: by source, then by length, then by
            # ascending rank id. Picking the lowest usable index therefore picks
            # the reference's path. A simple path in K4 visits distinct ranks, so
            # each hop uses a distinct pair class and the deltas never interact.
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
            improving = tl.sum(tl.where(usable, 1, 0), 0)
            split = split + tl.sum(tl.where(paths[:, None] == choice, delta, 0), 0)
            step = step + 1

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

    # This rank runs an active expert's hot copy when it owns the primary, and
    # its cold copy when the assignment picked this rank.
    hot_local = tl.load(hot_map_ptr + experts).to(tl.int32)
    cold_local = tl.load(cold_map_ptr + experts).to(tl.int32)
    hot_mine = active & (primary_hot != 0) & (primary == EP_RANK) & (hot_local >= 0)
    cold_mine = active & (primary_hot == 0) & (selected == EP_RANK) & (cold_local >= 0)

    # The rank of a route within its expert does not depend on the tier, since
    # an expert lives in exactly one of them, so it is computed once here.
    earlier = (
        (routed[:, None] == routed[None, :])
        & (routes[None, :] < routes[:, None])
        & route_mask[None, :]
    )
    within = tl.sum(earlier.to(tl.int32), 1)

    _emit_alignment(
        routed, route_mask, within, counts, hot_mine, hot_local, num_routes,
        scratch_ptr, hot_sorted_ptr, hot_expert_ids_ptr, hot_num_post_ptr,
        NUM_EXPERTS=NUM_EXPERTS, ROUTE_BLOCK=ROUTE_BLOCK, BLOCK_M=BLOCK_M_HOT,
        MAX_SORTED=MAX_SORTED_HOT, MAX_BLOCKS=MAX_BLOCKS_HOT,
    )
    tl.debug_barrier()
    _emit_alignment(
        routed, route_mask, within, counts, cold_mine, cold_local, num_routes,
        scratch_ptr, cold_sorted_ptr, cold_expert_ids_ptr, cold_num_post_ptr,
        NUM_EXPERTS=NUM_EXPERTS, ROUTE_BLOCK=ROUTE_BLOCK, BLOCK_M=BLOCK_M_COLD,
        MAX_SORTED=MAX_SORTED_COLD, MAX_BLOCKS=MAX_BLOCKS_COLD,
    )


def align_buffer_shapes(
    num_routes: int, num_experts: int, block_m: int
) -> tuple[int, int]:
    """Buffer sizes ``moe_align_block_size`` would allocate, rounded for Triton."""
    padded = num_routes + num_experts * (block_m - 1)
    if num_routes < num_experts:
        padded = min(num_routes * block_m, padded)
    padded = triton.next_power_of_2(padded)
    return padded, triton.next_power_of_2(-(-padded // block_m))


class TierMaps(NamedTuple):
    """Global-to-tier-local index per expert on one rank, -1 where absent."""

    hot: torch.Tensor
    cold: torch.Tensor


class FusedRouting(NamedTuple):
    scratch: torch.Tensor
    selected_rank: torch.Tensor
    hot_sorted: torch.Tensor
    hot_expert_ids: torch.Tensor
    hot_num_post: torch.Tensor
    cold_sorted: torch.Tensor
    cold_expert_ids: torch.Tensor
    cold_num_post: torch.Tensor


def fused_assign_align(
    topk_ids: torch.Tensor,
    primary_rank: torch.Tensor,
    secondary_rank: torch.Tensor,
    primary_hot: torch.Tensor,
    maps: TierMaps,
    ep_rank: int,
    block_m_hot: int = 16,
    block_m_cold: int = 16,
    schedule: bool = True,
    reversal_limit: int = MAX_REVERSALS,
    num_warps: int = 4,
    out: FusedRouting | None = None,
) -> FusedRouting:
    """Assign replicas and build both tiers' Marlin metadata in one launch."""
    num_experts = primary_rank.numel()
    flat = topk_ids.reshape(-1)
    device = flat.device
    num_routes = flat.numel()
    hot_sorted_len, hot_blocks = align_buffer_shapes(
        num_routes, num_experts, block_m_hot
    )
    cold_sorted_len, cold_blocks = align_buffer_shapes(
        num_routes, num_experts, block_m_cold
    )
    if out is None:
        def empty(size: int) -> torch.Tensor:
            return torch.empty(size, dtype=torch.int32, device=device)

        out = FusedRouting(
            empty(2 * num_experts + max(hot_blocks, cold_blocks)),
            empty(num_experts),
            empty(hot_sorted_len),
            empty(hot_blocks),
            empty(1),
            empty(cold_sorted_len),
            empty(cold_blocks),
            empty(1),
        )
    table = _path_table(device)
    _assign_kernel[(1,)](
        flat,
        primary_rank,
        secondary_rank,
        primary_hot,
        table[0],
        table[1],
        table[2],
        maps.hot,
        maps.cold,
        out.scratch,
        out.selected_rank,
        out.hot_sorted,
        out.hot_expert_ids,
        out.hot_num_post,
        out.cold_sorted,
        out.cold_expert_ids,
        out.cold_num_post,
        num_routes,
        NUM_EXPERTS=num_experts,
        ROUTE_BLOCK=triton.next_power_of_2(max(num_routes, 8)),
        PAIR_BLOCK=NUM_PAIRS_PADDED,
        PATH_BLOCK=triton.next_power_of_2(table[0].numel()),
        NUM_PATHS=table[0].numel(),
        EP_SIZE=EP,
        EP_RANK=ep_rank,
        SCHEDULE=schedule,
        REVERSAL_LIMIT=reversal_limit,
        BLOCK_M_HOT=block_m_hot,
        MAX_SORTED_HOT=hot_sorted_len,
        MAX_BLOCKS_HOT=hot_blocks,
        BLOCK_M_COLD=block_m_cold,
        MAX_SORTED_COLD=cold_sorted_len,
        MAX_BLOCKS_COLD=cold_blocks,
        num_warps=num_warps,
    )
    return out


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


def build_tier_maps(
    owners: np.ndarray,
    hot: np.ndarray,
    secondary: np.ndarray,
    ep_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Global-to-local maps for one rank's hot and cold tiers in one layer.

    Mirrors ``tiered_moe_storage.allocate_layer_expert_storage``: the cold tier
    holds this rank's primary-cold experts followed by its replica copies.
    """
    num_experts = owners.size
    hot_ids = sorted(
        e for e in range(num_experts) if owners[e] == ep_rank and hot[e]
    )
    cold_ids = sorted(
        e for e in range(num_experts) if owners[e] == ep_rank and not hot[e]
    )
    replica_ids = sorted(e for e in range(num_experts) if secondary[e] == ep_rank)
    hot_map = np.full(num_experts, -1, dtype=np.int32)
    cold_map = np.full(num_experts, -1, dtype=np.int32)
    hot_map[hot_ids] = np.arange(len(hot_ids), dtype=np.int32)
    cold_map[cold_ids + replica_ids] = np.arange(
        len(cold_ids) + len(replica_ids), dtype=np.int32
    )
    return hot_map, cold_map


def alignment_matches(
    mine: np.ndarray,
    reference: np.ndarray,
    counts: np.ndarray,
    mine_mask: np.ndarray,
    local: np.ndarray,
    block_m: int,
) -> bool:
    """Compare two alignments up to within-expert route order.

    vLLM's ``moe_align_block_size`` is not stable by flattened route index at
    decode sizes - its parallel counting sort emits an expert's routes in an
    arbitrary order - so an elementwise compare would fail on a correct kernel.
    Order within an expert has no effect on the GEMM, which treats each block as
    a set, or on ``moe_sum``, which indexes by route. What must match is the
    per-expert partition and the padding.
    """
    blocks = np.where(mine_mask, -(-counts // block_m), 0)
    # The kernel lays blocks out in ascending global expert id; the reference
    # uses tier-local index order. Compare each expert's route set in whichever
    # slice each side put it, since block order is not observable in the GEMM.
    mine_starts = np.cumsum(blocks) - blocks
    order = np.argsort(np.where(mine_mask, local, np.iinfo(np.int32).max))
    ref_starts = np.zeros_like(blocks)
    running = 0
    for expert in order:
        ref_starts[expert] = running
        running += blocks[expert]
    for expert in np.flatnonzero(blocks):
        span = int(blocks[expert]) * block_m
        mine_lo = int(mine_starts[expert]) * block_m
        ref_lo = int(ref_starts[expert]) * block_m
        if not np.array_equal(
            np.sort(mine[mine_lo : mine_lo + span]),
            np.sort(reference[ref_lo : ref_lo + span]),
        ):
            return False
    end = int(blocks.sum()) * block_m
    return np.array_equal(mine[end:], reference[end:])


def expected_block_experts(
    counts: np.ndarray,
    mine_mask: np.ndarray,
    local: np.ndarray,
    block_m: int,
    size: int,
) -> np.ndarray:
    """Tier-local expert index per block under the kernel's global-id layout."""
    blocks = np.where(mine_mask, -(-counts // block_m), 0)
    starts = np.cumsum(blocks) - blocks
    expected = np.full(size, -1, dtype=np.int32)
    for expert in np.flatnonzero(blocks):
        lo = int(starts[expert])
        expected[lo : lo + int(blocks[expert])] = local[expert]
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--concurrency", type=int, nargs="+", default=(1, 4))
    parser.add_argument("--verify-tokens", type=int, default=4)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

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
    num_experts = owners.shape[1]

    report: dict = {"placement": str(args.placement), "workloads": {}}
    for concurrency in args.concurrency:
        steps = sample_steps(heldout, args.steps, concurrency, rng)
        counts = route_counts(steps)
        failures = {
            "assignment": 0,
            "hot_sorted": 0,
            "hot_expert_ids": 0,
            "hot_num_post": 0,
            "cold_sorted": 0,
            "cold_expert_ids": 0,
            "cold_num_post": 0,
            "exactly_once": 0,
        }
        checked = 0
        for step in range(steps.shape[0]):
            for layer in range(NUM_LAYERS):
                routes = steps[step, :, layer, :].astype(np.int32)
                topk = torch.from_numpy(routes).to(device)
                primary_t = torch.from_numpy(owners[layer].astype(np.int32)).to(device)
                replica_t = torch.from_numpy(
                    secondary[layer].astype(np.int32)
                ).to(device)
                hot_t = torch.from_numpy(hot[layer].astype(np.int32)).to(device)

                expected_sel = reference_assign(
                    counts[step, layer].astype(np.int64),
                    owners[layer].astype(np.int64),
                    secondary[layer].astype(np.int64),
                    hot[layer].astype(np.int64),
                )
                executed = np.zeros(num_experts, dtype=np.int64)
                for ep_rank in range(EP):
                    hot_map, cold_map = build_tier_maps(
                        owners[layer], hot[layer], secondary[layer], ep_rank
                    )
                    maps = TierMaps(
                        torch.from_numpy(hot_map).to(device),
                        torch.from_numpy(cold_map).to(device),
                    )
                    got = fused_assign_align(
                        topk, primary_t, replica_t, hot_t, maps, ep_rank,
                        block_m_hot=args.block_m, block_m_cold=args.block_m,
                    )
                    checked += 1
                    selected = got.selected_rank.cpu().numpy().astype(np.int64)
                    if not np.array_equal(selected, expected_sel):
                        failures["assignment"] += 1

                    active = counts[step, layer] > 0
                    hot_mine = active & (hot[layer] != 0) & (owners[layer] == ep_rank)
                    cold_mine = (
                        active & (hot[layer] == 0) & (expected_sel == ep_rank)
                    )
                    executed += hot_mine.astype(np.int64)
                    executed += cold_mine.astype(np.int64)

                    for tier, mine, local, sorted_out, ids_out, post_out in (
                        ("hot", hot_mine, hot_map, got.hot_sorted,
                         got.hot_expert_ids, got.hot_num_post),
                        ("cold", cold_mine, cold_map, got.cold_sorted,
                         got.cold_expert_ids, got.cold_num_post),
                    ):
                        tier_map = np.where(mine, local, -1).astype(np.int32)
                        ref_sorted, ref_ids, ref_post = moe_align_block_size(
                            topk,
                            args.block_m,
                            num_experts,
                            torch.from_numpy(tier_map).to(device),
                            ignore_invalid_experts=True,
                        )
                        size = ref_sorted.numel()
                        if not alignment_matches(
                            sorted_out[:size].cpu().numpy(),
                            ref_sorted.cpu().numpy(),
                            counts[step, layer].astype(np.int64),
                            mine,
                            local,
                            args.block_m,
                        ):
                            failures[f"{tier}_sorted"] += 1
                        # expert_ids follows the kernel's global-id block order,
                        # so build the expectation rather than compare buffers.
                        expected_ids = expected_block_experts(
                            counts[step, layer].astype(np.int64),
                            mine,
                            local,
                            args.block_m,
                            ref_ids.numel(),
                        )
                        if not np.array_equal(
                            ids_out[: ref_ids.numel()].cpu().numpy(), expected_ids
                        ):
                            failures[f"{tier}_expert_ids"] += 1
                        if not torch.equal(post_out, ref_post):
                            failures[f"{tier}_num_post"] += 1

                if not np.array_equal(executed, active.astype(np.int64)):
                    failures["exactly_once"] += 1

        name = f"c{concurrency}"
        report["workloads"][name] = {
            "rank_layers_checked": checked,
            "failures": dict(failures),
        }
        total = sum(failures.values())
        print(
            f"{name}: {checked} rank-layers checked, {total} failures "
            + (
                "ALL PASS"
                if total == 0
                else str({k: v for k, v in failures.items() if v})
            ),
            flush=True,
        )

    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
