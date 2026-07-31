#!/usr/bin/env python3
"""Kernel-shaped min-max orientation solver, validated against max flow.

The fused kernel cannot run a max-flow. It must solve the layer assignment with
fixed-size integer state and bounded loops. This module implements the shape it
will use and checks it is *exactly* optimal on the real route distribution.

State is six pair-class counters ``x[c]`` = how many edges of class ``c`` are
oriented at the class's lower-numbered rank. Load of rank ``r`` is its fixed
offset plus the edges of every incident class pointing at it.

Algorithm (path reversal, exact for min-max in-degree orientation):

  1. orient every edge at its primary rank;
  2. while some rank ``u`` has load ``d`` and some rank ``v`` has load
     ``<= d - 2`` reachable from ``u`` by a path of classes that currently have
     an edge oriented the right way, reverse one edge along that path.

Each reversal strictly decreases the sum of squared loads, so it terminates.
With four ranks every path has length at most three, so the search is three
fixed nested loops.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from replay_exact import (
    EP,
    NUM_LAYERS,
    PAIRS,
    PAIR_INDEX,
    load_placement,
    load_requests,
    route_counts,
    sample_steps,
    solve_min_max,
)

MAX_REVERSALS = 256
ALL_SOURCES = bool(int(os.environ.get("REPLICA_ALL_SOURCES", "0")))


def orient_path_reversal(
    offsets: np.ndarray, pair_counts: np.ndarray, at_low: np.ndarray
) -> tuple[np.ndarray, int]:
    """Return an optimal split and the number of reversals used.

    ``at_low[c]`` is the initial number of class-``c`` edges oriented at the
    class's lower rank, i.e. the primary-only orientation.
    """
    split = at_low.astype(np.int64).copy()

    def rank_loads() -> np.ndarray:
        result = offsets.astype(np.int64).copy()
        for index, (low, high) in enumerate(PAIRS):
            result[low] += split[index]
            result[high] += pair_counts[index] - split[index]
        return result

    def movable(source: int, target: int) -> bool:
        """Is there an edge between these ranks currently pointing at source?"""
        if source == target:
            return False
        index = PAIR_INDEX[(min(source, target), max(source, target))]
        return split[index] > 0 if source < target else (
            pair_counts[index] - split[index] > 0
        )

    def move(source: int, target: int) -> None:
        index = PAIR_INDEX[(min(source, target), max(source, target))]
        if source < target:
            split[index] -= 1
        else:
            split[index] += 1

    reversals = 0
    path_lengths: list[int] = []
    while reversals < MAX_REVERSALS:
        load = rank_loads()
        # Deterministic: heaviest rank, lowest id first; then the shortest,
        # lowest-id path to a rank at least two lighter.
        path = None
        # The classical optimality condition only concerns maximum-load
        # vertices, so the kernel searches from the lowest-id maximum. Set
        # REPLICA_ALL_SOURCES=1 to search every rank in descending-load order.
        candidates = (
            np.argsort(-load, kind="stable")
            if ALL_SOURCES
            else [int(np.argmax(load))]
        )
        for source in candidates:
            source = int(source)
            threshold = int(load[source]) - 2
            for length in (1, 2, 3):
                path = _find_path(source, length, threshold, load, movable, [source])
                if path is not None:
                    break
            if path is not None:
                break
        if path is None:
            break
        path_lengths.append(len(path) - 1)
        for step in range(len(path) - 1):
            move(path[step], path[step + 1])
        reversals += 1
    else:
        raise RuntimeError("Path reversal did not converge")
    return split, reversals, path_lengths


def _find_path(
    source: int,
    length: int,
    threshold: int,
    load: np.ndarray,
    movable,
    visited: list[int],
) -> list | None:
    """Lowest-id simple path of exactly ``length`` hops to a rank at ``threshold``.

    Reversing every edge along the path moves one unit of load from the path's
    first rank to its last; every intermediate rank gains one and loses one, so
    only the endpoints change and only they need checking.
    """
    if length == 1:
        for target in range(EP):
            if target in visited:
                continue
            if load[target] <= threshold and movable(source, target):
                return [source, target]
        return None
    for middle in range(EP):
        if middle in visited or not movable(source, middle):
            continue
        tail = _find_path(
            middle, length - 1, threshold, load, movable, visited + [middle]
        )
        if tail is not None:
            return [source] + tail
    return None


def layer_problem(
    counts: np.ndarray,
    owners: np.ndarray,
    hot: np.ndarray,
    secondary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Reduce one layer to offsets, pair counts and the primary orientation."""
    active = np.flatnonzero(counts)
    if active.size == 0:
        return None
    primary = owners[active].astype(np.int64)
    is_hot = hot[active]
    replica = secondary[active].astype(np.int64)

    offsets = np.zeros(EP, dtype=np.int64)
    pair_counts = np.zeros(len(PAIRS), dtype=np.int64)
    at_low = np.zeros(len(PAIRS), dtype=np.int64)
    for index in range(active.size):
        if is_hot[index]:
            continue
        rank = int(primary[index])
        target = int(replica[index])
        if target < 0:
            offsets[rank] += 1
            continue
        slot = PAIR_INDEX[(min(rank, target), max(rank, target))]
        pair_counts[slot] += 1
        if rank < target:
            at_low[slot] += 1
    return offsets, pair_counts, at_low


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--c1-steps", type=int, default=100)
    parser.add_argument("--c4-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    heldout = load_requests(args.trace_dir, "heldout")
    owners, hot, secondary = load_placement(args.placement)

    report: dict = {"placement": str(args.placement), "workloads": {}}
    for name, concurrency, steps in (
        ("c1", 1, args.c1_steps),
        ("c4", 4, args.c4_steps),
    ):
        counts = route_counts(sample_steps(heldout, steps, concurrency, rng))
        solved = 0
        optimal = 0
        reversal_hist: dict[int, int] = {}
        length_hist: dict[int, int] = {}
        worst_gap = 0
        for step in range(counts.shape[0]):
            for layer in range(NUM_LAYERS):
                problem = layer_problem(
                    counts[step, layer], owners[layer], hot[layer], secondary[layer]
                )
                if problem is None:
                    continue
                offsets, pair_counts, at_low = problem
                if pair_counts.sum() == 0:
                    continue
                split, reversals, lengths = orient_path_reversal(
                    offsets, pair_counts, at_low
                )
                for length in lengths:
                    length_hist[length] = length_hist.get(length, 0) + 1

                load = offsets.copy()
                for index, (low, high) in enumerate(PAIRS):
                    load[low] += split[index]
                    load[high] += pair_counts[index] - split[index]
                achieved = int(load.max())
                best, _ = solve_min_max(offsets, pair_counts, cross_check=False)

                solved += 1
                optimal += achieved == best
                worst_gap = max(worst_gap, achieved - best)
                reversal_hist[reversals] = reversal_hist.get(reversals, 0) + 1

        report["workloads"][name] = {
            "layers_solved": solved,
            "optimal": optimal,
            "optimal_percent": 100.0 * optimal / max(solved, 1),
            "worst_gap": worst_gap,
            "max_reversals": max(reversal_hist) if reversal_hist else 0,
            "mean_reversals": (
                sum(k * v for k, v in reversal_hist.items()) / max(solved, 1)
            ),
            "reversal_histogram": {str(k): v for k, v in sorted(reversal_hist.items())},
            "path_length_histogram": {
                str(k): v for k, v in sorted(length_hist.items())
            },
        }
        row = report["workloads"][name]
        print(
            f"{name}: {row['optimal']}/{row['layers_solved']} optimal "
            f"({row['optimal_percent']:.4f}%), worst gap {row['worst_gap']}, "
            f"reversals mean {row['mean_reversals']:.2f} max {row['max_reversals']}, "
            f"path lengths {row['path_length_histogram']}"
        )

    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
