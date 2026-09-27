#!/usr/bin/env python3
"""Break down MiMo-V2.6 decode steps from torch-profiler traces.

Kernels are attributed to a step by launch correlation (the CPU launch that
produced them), not by where their GPU timestamps fall. Within a step:

  target     the verify forward: the one large CUDA graph (~2.3K kernels)
  logits     lm_head GEMM and vocabulary all-gather launched right after it
  draft      the DFlash drafter: small graphs plus eager attention
  host-side  everything else launched from the step (input prep, sampling)

Time inside a phase is split with the union-consistent partition from
2026-07-29-marlin-smem-monopoly/analyze_step_budget.py: each elementary
segment is shared equally among the categories active in it, so category
shares sum to the phase's busy time. Hot and cold Marlin are told apart by
grid size (hot 264 CTAs = 2/SM, cold 132 = 1/SM, the shipped launch policy).

Collectives are also reported cross-rank: for each (step, ordinal) the
minimum duration over the four ranks approximates the transfer itself, and the
excess over it is time spent waiting for the slowest rank to arrive.

    analyze.py <trace-dir>... [--json out.json]
"""

import argparse
import bisect
import collections
import gzip
import json
import re
import statistics
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
RANK_RE = re.compile(r"_rank(\d+)\.")

CATEGORIES = (
    # the one-kernel tiered decode path (VLLM_TIERED_MOE_DECODE_KERNEL=1):
    # both tiers inside one launch per projection
    ("MoE one-kernel w13", lambda e: "tiered_decode" in e["name"] and "gemm_kernel<0>" in e["name"]),
    ("MoE one-kernel w2", lambda e: "tiered_decode" in e["name"] and "gemm_kernel<1>" in e["name"]),
    ("MoE one-kernel route/act/finalize", lambda e: "tiered_decode" in e["name"]),
    ("MoE hot Marlin (HBM)", lambda e: "marlin_moe" in e["name"] and grid(e) == 264),
    ("MoE cold Marlin (Grace)", lambda e: "marlin_moe" in e["name"]),
    ("MoE reduce-scatter (SP)", lambda e: "ReduceScatter" in e["name"]),
    ("MoE all-gather (SP)", lambda e: "AllGather" in e["name"]),
    ("TP all-reduce", lambda e: "cross_device_reduce" in e["name"]),
    ("MoE routing/align/sum/act", lambda e: any(k in e["name"] for k in (
        "grouped_topk", "moe_align", "count_and_sort", "moe_sum", "act_and_mul",
        "_assign_kernel", "_route_fingerprint"))),
    ("attention", lambda e: any(k in e["name"] for k in (
        "flash", "Flash", "reshape_and_cache", "prepare_varlen", "slot_mapping"))),
    ("dense GEMM", lambda e: any(k in e["name"] for k in (
        "nvjet", "deep_gemm", "cutlass", "splitKreduce", "fp8_blockscale", "gemm"))),
    ("norm/rope/elementwise", lambda e: True),
)


def grid(event: dict) -> int:
    value = event["args"].get("grid") or [0]
    return value[0] if isinstance(value, list) else int(value)


def category(event: dict) -> str:
    for label, test in CATEGORIES:
        if test(event):
            return label
    raise AssertionError


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        events = [e for e in json.load(handle)["traceEvents"] if e.get("ph") == "X"]
    for event in events:
        event["t"] = event["ts"] / 1000
        event["end"] = event["t"] + event.get("dur", 0) / 1000
    return events


def union(events: list[dict]) -> float:
    total, cursor = 0.0, None
    for event in sorted(events, key=lambda e: e["t"]):
        if cursor is None or event["t"] > cursor:
            total += event["end"] - event["t"]
            cursor = event["end"]
        elif event["end"] > cursor:
            total += event["end"] - cursor
            cursor = event["end"]
    return total


def partition(events: list[dict]) -> dict[str, float]:
    edges = sorted({x for e in events for x in (e["t"], e["end"])})
    spans = sorted(((e["t"], e["end"], category(e)) for e in events), key=lambda s: s[0])
    shares = collections.defaultdict(float)
    active, cursor = [], 0
    for left, right in zip(edges, edges[1:]):
        while cursor < len(spans) and spans[cursor][0] <= left:
            active.append(spans[cursor])
            cursor += 1
        active = [s for s in active if s[1] > left]
        names = {s[2] for s in active}
        for name in names:
            shares[name] += (right - left) / len(names)
    return dict(shares)


def steps(events: list[dict]) -> list[dict]:
    by_corr = collections.defaultdict(list)
    for event in events:
        if event.get("cat") in GPU_CATS:
            by_corr[event["args"].get("correlation")].append(event)
    windows = sorted(
        (e for e in events if e.get("cat") == "user_annotation"
         and e["name"].startswith("execute_")), key=lambda e: e["t"])
    starts = [w["t"] for w in windows]
    launches = sorted((e for e in events if e.get("cat") in LAUNCH_CATS), key=lambda e: e["t"])
    per_step = collections.defaultdict(list)
    for launch in launches:
        index = bisect.bisect(starts, launch["t"]) - 1
        if 0 <= index < len(windows) - 1:
            per_step[index].append(launch)
    out = []
    for index in sorted(per_step):
        phases = collections.defaultdict(list)
        graphs = [l for l in per_step[index] if "GraphLaunch" in l["name"]]
        target = max(graphs, key=lambda l: len(by_corr[l["args"]["correlation"]]))
        after_target = False
        for launch in per_step[index]:
            ops = by_corr.get(launch["args"].get("correlation"), [])
            if not ops:
                continue
            if launch is target:
                phases["target"] += ops
                after_target = True
            elif "GraphLaunch" in launch["name"]:
                phases["draft"] += ops
            elif after_target and not phases["draft"] and any(
                    "nvjet" in o["name"] or "AllGather" in o["name"] for o in ops):
                phases["logits"] += ops
            elif phases["draft"] or any("dflash" in o["name"] for o in ops):
                phases["draft"] += ops
            else:
                phases["host-side"] += ops
        everything = [o for ops in phases.values() for o in ops]
        out.append({"phases": phases, "span": (min(o["t"] for o in everything),
                                               max(o["end"] for o in everything))})
    for this, nxt in zip(out, out[1:]):
        this["period"] = nxt["span"][0] - this["span"][0]
    return out[:-1]


def collectives(step_rows: list[dict]) -> dict[tuple, float]:
    """(step, name, ordinal) -> duration, for the target graph's collectives."""
    table = {}
    for index, row in enumerate(step_rows):
        counts = collections.Counter()
        for op in sorted(row["phases"]["target"], key=lambda e: e["t"]):
            for key in ("ReduceScatter", "AllGather", "cross_device_reduce"):
                if key in op["name"]:
                    table[(index, key, counts[key])] = op["end"] - op["t"]
                    counts[key] += 1
    return table


def analyze(trace_dir: Path) -> dict:
    ranks = {}
    for path in sorted(trace_dir.glob("*.pt.trace.json.gz")):
        rank = int(RANK_RE.search(path.name).group(1))
        ranks[rank] = steps(load(path))
    result = {"ranks": {}, "collectives": {}}
    for rank, rows in ranks.items():
        period = [r["period"] for r in rows]
        phase_busy = collections.defaultdict(list)
        cats = collections.defaultdict(list)
        marlin = collections.defaultdict(list)
        for row in rows:
            for name, ops in row["phases"].items():
                phase_busy[name].append(union(ops))
            target = row["phases"]["target"]
            for name, value in partition(target).items():
                cats[name].append(value)
            hot = [o for o in target if category(o) == "MoE hot Marlin (HBM)"]
            cold = [o for o in target if category(o) == "MoE cold Marlin (Grace)"]
            marlin["hot sum"].append(sum(o["end"] - o["t"] for o in hot))
            marlin["cold sum"].append(sum(o["end"] - o["t"] for o in cold))
            marlin["union"].append(union(hot + cold))
            span = max(o["end"] for o in target) - min(o["t"] for o in target)
            marlin["target span"].append(span)
        mean = lambda xs: statistics.fmean(xs) if xs else 0.0  # noqa: E731
        result["ranks"][rank] = {
            "steps": len(rows),
            "period_ms": {"mean": mean(period), "p50": statistics.median(period)},
            "phase_busy_ms": {k: mean(v) for k, v in phase_busy.items()},
            "target_categories_ms": {k: sum(v) / len(rows) for k, v in cats.items()},
            "marlin_ms": {k: mean(v) for k, v in marlin.items()},
        }
    tables = {rank: collectives(rows) for rank, rows in ranks.items()}
    common = set.intersection(*(set(t) for t in tables.values()))
    by_kind = collections.defaultdict(lambda: {"mean": [], "min": []})
    for key in common:
        values = [tables[rank][key] for rank in tables]
        by_kind[key[1]]["mean"].append(statistics.fmean(values))
        by_kind[key[1]]["min"].append(min(values))
    n_steps = min(len(rows) for rows in ranks.values())
    for kind, lists in by_kind.items():
        result["collectives"][kind] = {
            "per_step": len(lists["mean"]) / n_steps,
            "mean_ms_per_step": sum(lists["mean"]) / n_steps,
            "cross_rank_min_ms_per_step": sum(lists["min"]) / n_steps,
        }
    return result


def report(label: str, result: dict) -> None:
    ranks = result["ranks"]
    avg = lambda f: statistics.fmean(f(r) for r in ranks.values())  # noqa: E731
    print(f"\n===== {label}: {len(ranks)} ranks, {avg(lambda r: r['steps']):.0f} steps each")
    print(f"step period (GPU start to start): {avg(lambda r: r['period_ms']['mean']):.2f} ms"
          f"  per rank {[round(r['period_ms']['mean'], 2) for r in ranks.values()]}")
    print("phase busy time (ms):", {k: round(avg(lambda r: r['phase_busy_ms'].get(k, 0)), 3)
                                    for k in ("target", "logits", "draft", "host-side")})
    period = avg(lambda r: r["period_ms"]["mean"])
    busy = avg(lambda r: sum(r["phase_busy_ms"].values()))
    print(f"GPU idle inside the step: {period - busy:.2f} ms")
    print("\ntarget verify graph, union-partitioned (ms/step, mean over ranks; min-max rank):")
    names = [label for label, _ in CATEGORIES]
    total = avg(lambda r: sum(r["target_categories_ms"].values()))
    for name in names:
        values = [r["target_categories_ms"].get(name, 0) for r in ranks.values()]
        print(f"  {name:28s} {statistics.fmean(values):6.2f}  ({min(values):.2f}-{max(values):.2f})"
              f"  {statistics.fmean(values) / total * 100:5.1f}%")
    print(f"  {'target busy':28s} {total:6.2f}")
    m = {k: avg(lambda r, k=k: r["marlin_ms"][k]) for k in ranks[0]["marlin_ms"]}
    print(f"\nMarlin: hot sum {m['hot sum']:.2f}, cold sum {m['cold sum']:.2f}, "
          f"union {m['union']:.2f} ms/step (serial would be {m['hot sum'] + m['cold sum']:.2f}); "
          f"target span {m['target span']:.2f}")
    print("\ncollectives in the target graph (ms/step): mean over ranks vs cross-rank min")
    for kind, row in result["collectives"].items():
        print(f"  {kind:22s} x{row['per_step']:5.1f}  mean {row['mean_ms_per_step']:6.2f}  "
              f"min {row['cross_rank_min_ms_per_step']:6.2f}  -> waiting "
              f"{row['mean_ms_per_step'] - row['cross_rank_min_ms_per_step']:5.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = {}
    for directory in args.dirs:
        results[directory.name] = analyze(directory)
        report(directory.name, results[directory.name])
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
