#!/usr/bin/env python3
"""Deep dive into the GLM target/verify CUDA graph (newest traces).

Below analyze.py's category buckets: per-kernel solo/critical-path time
(the union segment where exactly one kernel is active), internal graph
idle with recurring sites, stream concurrency, the per-layer repeating
pattern, tiered-MoE chain overlap structure, and cross-rank collective
waiting per ordinal.

    verify_graph_dive.py <trace-dir>... [--rank 0] [--ordinals] [--layers]
"""

import argparse
import collections
import importlib.util
import re
import statistics
import sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "analyze", Path(__file__).resolve().parent.parent
    / "2026-09-26-mimo-decode-profile/analyze.py")
analyze = importlib.util.module_from_spec(SPEC)
sys.modules["analyze"] = analyze
SPEC.loader.exec_module(analyze)

RANK_RE = re.compile(r"_rank(\d+)\.")
LAYER_MARK = "fused_add_rms_norm_2"  # one per target layer


def display_name(op):
    name = op["name"]
    if "nvjet" in name or "splitKreduce" in name:
        grid = op["args"].get("grid")
        g = grid[0] if isinstance(grid, list) else grid
        name = f"{name[:60]}[g{g}]"
    return name


def solo_time(steps_, n_top=18):
    """Time each kernel is the only one running (its critical-path cost)."""
    solo = collections.defaultdict(float)
    joint = collections.defaultdict(float)
    for row in steps_:
        ops = row["phases"]["target"]
        edges = sorted({x for o in ops for x in (o["t"], o["end"])})
        active, cursor = [], 0
        spans = sorted((o["t"], o["end"], display_name(o)) for o in ops)
        for left, right in zip(edges, edges[1:]):
            while cursor < len(spans) and spans[cursor][0] <= left:
                active.append(spans[cursor])
                cursor += 1
            active = [s for s in active if s[1] > left]
            names = {s[2] for s in active}
            for name in names:
                if len(names) == 1:
                    solo[name] += (right - left)
                else:
                    joint[name] += (right - left) / len(names)
    span = len(steps_)
    total_solo = sum(solo.values()) / span
    print(f"\nsolo (critical-path) time per kernel, ms/step. Total solo "
          f"{total_solo:.2f} ms of the graph; top:")
    for name, t in sorted(solo.items(), key=lambda kv: -kv[1])[:n_top]:
        print(f"  {t / span:7.3f}   {name[:96]}")
    print(f"\nmost-hidden kernels (joint-overlap share, ms/step; near-zero solo):")
    rows = [(joint[n] / span, solo.get(n, 0) / span, n)
            for n in joint if solo.get(n, 0) / span < 0.15]
    for j, s, name in sorted(rows, key=lambda r: -r[0])[:8]:
        print(f"  joint {j:7.3f}  solo {s:6.3f}   {name[:88]}")


def kernel_table(steps_, n_top=30):
    per = collections.defaultdict(lambda: [0.0, 0])
    span = len(steps_)
    for row in steps_:
        for op in row["phases"]["target"]:
            slot = per[display_name(op)[:100]]
            slot[0] += op["dur"] / 1000
            slot[1] += 1
    total = sum(v[0] for v in per.values()) / span
    print(f"\nkernel launches inside the graph (sum {total:.2f} ms/step over "
          f"{sum(v[1] for v in per.values()) / span:.0f} launches):")
    for name, (ms, count) in sorted(per.items(), key=lambda kv: -kv[1][0])[:n_top]:
        mean_us = ms / count * 1000 if count else 0
        print(f"  {ms / span:7.3f} ms {ms / span / total * 100:5.1f}%  "
              f"x{count / span:6.1f}  {mean_us:6.1f} us  {name[:92]}")


def idle_analysis(steps_, n_worst=8):
    idle_total = []
    gap_sites = collections.defaultdict(float)
    for row in steps_:
        ops = sorted(row["phases"]["target"], key=lambda op: op["t"])
        cursor = prev_name = None
        idle = 0.0
        for op in ops:
            if cursor is not None and op["t"] > cursor:
                idle += op["t"] - cursor
                gap_sites[(prev_name[:40], op["name"][:40])] += op["t"] - cursor
            if cursor is None or op["end"] > cursor:
                prev_name = op["name"]
                cursor = op["end"] if cursor is None else max(cursor, op["end"])
        idle_total.append(idle)
    print(f"\nGPU idle strictly inside the target graph span: "
          f"{statistics.fmean(idle_total):.3f} ms/step "
          f"(p50 {statistics.median(idle_total):.3f})")
    print("recurring internal gaps:")
    for (a, b), gap in sorted(gap_sites.items(), key=lambda kv: -kv[1])[:n_worst]:
        print(f"  {gap / len(steps_):6.3f} ms/step  after {a}  ->  {b}")


def stream_analysis(steps_):
    per = collections.defaultdict(float)
    count = collections.defaultdict(int)
    names = collections.defaultdict(collections.Counter)
    span = len(steps_)
    for row in steps_:
        for op in row["phases"]["target"]:
            s = op["args"].get("stream")
            per[s] += op["dur"] / 1000
            count[s] += 1
            names[s][op["name"][:56]] += 1
    union_per = [analyze.union(r["phases"]["target"]) for r in steps_]
    print("\nstreams inside the target graph:")
    for s in sorted(per, key=lambda s: -per[s]):
        print(f"  stream {s}: {per[s] / span:7.3f} ms  x{count[s] / span:.0f} launches")
        for name, n in names[s].most_common(3):
            print(f"      x{n / span:6.1f}  {name}")
    print(f"  union busy: {statistics.fmean(union_per):.3f} ms/step")


def kind_of(op):
    if "gemm_kernel<0" in op["name"]:
        return "w13"
    if "gemm_kernel<1" in op["name"]:
        return "w2"
    if "act_kernel" in op["name"]:
        return "act"
    if "finalize" in op["name"]:
        return "fin"
    return "prep"


def tiered_overlap(steps_):
    """How the tiered chain's act/finalize kernels relate to its GEMMs."""
    totals = collections.defaultdict(float)
    rel = collections.defaultdict(list)
    seq = collections.Counter()
    span = len(steps_)
    for row in steps_:
        tiers = sorted((op for op in row["phases"]["target"]
                        if "tiered_decode" in op["name"]), key=lambda op: op["t"])
        if not tiers:
            continue
        pairs = [(kind_of(o), o) for o in tiers]
        for kind, op in pairs:
            totals[kind] += op["dur"] / 1000
        seq[tuple(k for k, _ in pairs[:6])] += 1
        for (k1, o1), (k2, o2) in zip(pairs, pairs[1:]):
            overlap = max(0.0, min(o1["end"], o2["end"]) - max(o1["t"], o2["t"]))
            rel[(k1, k2)].append((o2["t"] - o1["t"], o2["end"] - o2["t"], overlap))
    print("\ntiered chain, ms of kernel time per step:",
          {k: round(v / span, 3) for k, v in sorted(totals.items())})
    for kinds, count in seq.most_common(2):
        print(f"  launch pattern {kinds} x{count / span:.0f}/step")
    for (k1, k2), rows in sorted(rel.items()):
        gaps = [g for g, e, o in rows]
        durs = [e for g, e, o in rows]
        ovs = [o for g, e, o in rows]
        n_over = sum(1 for o in ovs if o > 0.001)
        print(f"  {k1} -> {k2}: x{len(rows) / span:.0f}/step  next-start delta "
              f"{statistics.fmean(gaps) * 1000:6.1f} us  next-dur "
              f"{statistics.fmean(durs) * 1000:6.1f} us  overlap "
              f"{statistics.fmean(ovs) * 1000:6.1f} us  ({n_over / max(1, len(rows)) * 100:.0f}% of pairs)")


def layer_pattern(steps_, max_pos=44):
    """Aggregate the deterministic per-layer kernel sequence across steps."""
    seqs = []
    for row in steps_:
        ops = sorted(row["phases"]["target"], key=lambda op: op["t"])
        marks = [i for i, op in enumerate(ops) if LAYER_MARK in op["name"]]
        for a, b in zip(marks, marks[1:]):
            seqs.append(ops[a:b])
    length = min(len(s) for s in seqs)
    print(f"\nper-layer kernel sequence ({len(seqs)} layer instances pooled, "
          f"{length} kernels from the layer norm):")
    t0_durs = collections.defaultdict(list)
    for pos in range(min(length, max_pos)):
        durs = [s[pos]["dur"] for s in seqs]
        modal = collections.Counter(display_name(s[pos])[:84] for s in seqs).most_common(1)[0][0]
        print(f"  #{pos:3d}  mean {statistics.fmean(durs):6.1f}  p90 "
              f"{sorted(durs)[int(len(durs) * 0.9)]:6.1f} us  {modal}")


def collectives_by_ordinal(rank_steps):
    tables = {rank: analyze.collectives(rows) for rank, rows in rank_steps.items()}
    if len(tables) < 2:
        return
    common = set.intersection(*(set(t) for t in tables.values()))
    by_ord = collections.defaultdict(lambda: {"mean": [], "min": []})
    for key in common:
        values = [tables[r][key] for r in sorted(tables)]
        by_ord[key[2]]["mean"].append(statistics.fmean(values))
        by_ord[key[2]]["min"].append(min(values))
    print("\ncross_device_reduce waiting per ordinal (worst 15):")
    ranked = []
    for ordinal, d in by_ord.items():
        mean = statistics.fmean(d["mean"])
        vmin = statistics.fmean(d["min"])
        ranked.append((mean - vmin, ordinal, mean, vmin))
    for wait, ordinal, mean, vmin in sorted(ranked, reverse=True)[:15]:
        print(f"  #{ordinal:3d}  mean {mean:6.3f}  min {vmin:6.3f}  wait {wait:6.3f}")
    print(f"  total waiting over {len(ranked)} ordinals: "
          f"{sum(w for w, *_ in ranked if w > 0):.2f} ms/step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--layers", action="store_true")
    ap.add_argument("--ordinals", action="store_true")
    args = ap.parse_args()

    rank_steps = {}
    for d in args.dirs:
        for path in sorted(d.glob("*.pt.trace.json.gz")):
            rank = int(RANK_RE.search(path.name).group(1))
            rank_steps.setdefault(rank, []).extend(analyze.steps(analyze.load(path)))
    n = min(len(v) for v in rank_steps.values())
    for v in rank_steps.values():
        del v[n:]
    print(f"pooled {len(args.dirs)} window(s): {n} steps per rank; "
          f"ranks {sorted(rank_steps)}")

    steps_ = rank_steps[args.rank]
    periods = [r["period"] for r in steps_]
    target_busy = [analyze.union(r["phases"]["target"]) for r in steps_]
    print(f"step period {statistics.fmean(periods):.2f} ms, target busy "
          f"{statistics.fmean(target_busy):.2f} ms")
    for phase, label in (("target", "target/verify graph"), ("logits", "logits"),
                          ("draft", "drafter"), ("host-side", "host-side kernels")):
        busy = [analyze.union(r["phases"][phase]) for r in steps_ if r["phases"].get(phase)]
        if busy:
            print(f"  {label:22s} {statistics.fmean(busy):6.2f} ms")

    kernel_table(steps_, args.top)
    solo_time(steps_)
    idle_analysis(steps_)
    stream_analysis(steps_)
    tiered_overlap(steps_)
    if args.layers:
        layer_pattern(steps_)
    if args.ordinals:
        collectives_by_ordinal(rank_steps)


if __name__ == "__main__":
    main()
