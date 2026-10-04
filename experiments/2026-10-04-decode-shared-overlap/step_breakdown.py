#!/usr/bin/env python3
"""Whole served decode step under the production config (trace-pdcp2): phases,
union-partitioned kernel categories, cross-rank collective waiting, the
decode-MoE window per step, and how much of the aux-stream shared expert is
exposed. Extends ../2026-09-26-mimo-decode-profile/analyze.py to this build
(fused AR+RMS, one-shot DCP ops, tiered_decode one-kernel MoE, cute-dsl router).

    step_breakdown.py <trace-dir> [--json out.json]
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
DECODE_WINDOW = "execute_context_0(0)_generation_1(8)"

CATEGORIES = (
    ("MoE w13 (one-kernel)", lambda n: "tiered_decode" in n and "gemm_kernel<0" in n),
    ("MoE w2 (one-kernel)", lambda n: "tiered_decode" in n and "gemm_kernel<1" in n),
    ("MoE act/route_prep/finalize", lambda n: "tiered_decode" in n),
    ("router GEMM (cute-dsl)", lambda n: "cute_dsl_ll_bf16" in n),
    ("MoE routing (topk/align/sum)", lambda n: any(k in n for k in (
        "grouped_topk", "moe_align", "count_and_sort", "moe_sum", "act_and_mul",
        "_assign_kernel", "_route_fingerprint"))),
    ("DCP one-shot collectives", lambda n: "one_shot::" in n),
    ("TP all-reduce (fused AR+RMS)", lambda n: "allreduce_fusion" in n),
    ("TP all-reduce (custom)", lambda n: "cross_device_reduce" in n),
    ("DSA indexer", lambda n: any(k in n for k in (
        "mqa_logits", "topKPerRow", "top_k_per_row", "indexer", "indexer_k_quant",
        "convert_req_index", "pack_dcp_topk", "cooperative_topk"))),
    ("attention (sparse MLA)", lambda n: any(k in n for k in (
        "flash", "Flash", "MLA", "mla", "reshape_and_cache", "prepare_varlen",
        "slot_mapping"))),
    ("dense GEMM", lambda n: any(k in n for k in (
        "nvjet", "splitKreduce", "deep_gemm", "cutlass", "gemm"))),
    ("norm/rope/elementwise", lambda n: True),
)
COLLECTIVE_KEYS = ("one_shot::gather_cat", "one_shot::lse_reduce_scatter",
                   "one_shot::gather_kernel", "allreduce_fusion",
                   "cross_device_reduce")


def load(path: Path) -> list[dict]:
    events = [e for e in json.load(gzip.open(path, "rt"))["traceEvents"]
              if e.get("ph") == "X"]
    for event in events:
        event["t"] = event["ts"]
        event["end"] = event["ts"] + event.get("dur", 0)
    return events


def category(name: str) -> str:
    for label, test in CATEGORIES:
        if test(name):
            return label
    raise AssertionError(name)


def union(events) -> float:
    total, cursor = 0.0, None
    for event in sorted(events, key=lambda e: e["t"]):
        if cursor is None or event["t"] > cursor:
            total += event["end"] - event["t"]
            cursor = event["end"]
        elif event["end"] > cursor:
            total += event["end"] - cursor
            cursor = event["end"]
    return total


def intersect(a_events, b_union_events) -> float:
    """Total time of a_events covered by the union of b_union_events, in us."""
    b = []
    total, cursor = 0.0, None
    for event in sorted(b_union_events, key=lambda e: e["t"]):
        if cursor is None or event["t"] > cursor:
            b.append((event["t"], event["end"]))
            cursor = event["end"]
        elif event["end"] > cursor:
            b[-1] = (b[-1][0], event["end"])
            cursor = event["end"]
    starts = [x[0] for x in b]
    for event in a_events:
        i = bisect.bisect_right(starts, event["t"]) - 1
        t = event["t"]
        while i >= 0 and i < len(b):
            if b[i][1] <= t:
                if i + 1 < len(b) and b[i + 1][0] > event["end"]:
                    break
                i += 1
                continue
            total += min(b[i][1], event["end"]) - max(b[i][0], t)
            if b[i][1] >= event["end"]:
                break
            t = b[i][1]
            i += 1
    return total


def steps(events: list[dict]) -> list[dict]:
    by_corr = collections.defaultdict(list)
    for event in events:
        if event.get("cat") in GPU_CATS:
            by_corr[event["args"].get("correlation")].append(event)
    windows = sorted((e for e in events if e.get("cat") == "user_annotation"
                      and e["name"].startswith("execute_")), key=lambda e: e["t"])
    decode_idx = [i for i, w in enumerate(windows) if w["name"] == DECODE_WINDOW]
    starts = [w["t"] for w in windows]
    launches = sorted((e for e in events if e.get("cat") in LAUNCH_CATS), key=lambda e: e["t"])
    per_step = collections.defaultdict(list)
    for launch in launches:
        index = bisect.bisect(starts, launch["t"]) - 1
        per_step[index].append(launch)
    out = []
    for index in decode_idx:
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
        if "target" not in phases:
            continue
        everything = [o for ops in phases.values() for o in ops]
        out.append({"phases": phases, "span": (min(o["t"] for o in everything),
                                               max(o["end"] for o in everything))})
    for this, nxt in zip(out, out[1:]):
        this["period"] = nxt["span"][0] - this["span"][0]
    return out[:-1]


def partition(events: list[dict]) -> dict[str, float]:
    edges = sorted({x for e in events for x in (e["t"], e["end"])})
    spans = sorted(((e["t"], e["end"], category(e["name"])) for e in events), key=lambda s: s[0])
    shares = collections.defaultdict(float)
    active, cursor = [], 0
    for left, right in zip(edges, edges[1:]):
        while cursor < len(spans) and spans[cursor][0] <= left:
            active.append(spans[cursor])
            cursor += 1
        active = [s for s in active if s[1] > left]
        for name in {s[2] for s in active}:
            shares[name] += (right - left) / len({s[2] for s in active})
    return dict(shares)


def collectives(step_rows: list[dict]) -> dict:
    table = {}
    for index, row in enumerate(step_rows):
        counts = collections.Counter()
        prev = None
        for op in sorted(row["phases"]["target"], key=lambda e: e["t"]):
            key = next((k for k in COLLECTIVE_KEYS if k in op["name"]), None)
            if key:
                cls = ""
                if key == "allreduce_fusion" and prev is not None:
                    p = prev["name"]
                    cls = "" if "nvjet" in p else ".post-MoE"  # o_proj vs fused_add_mul
                table[(index, key + cls, counts[(key, cls)])] = op["end"] - op["t"]
                counts[(key, cls)] += 1
            prev = op
    return table


def analyze(trace_dir: Path) -> dict:
    ranks = {}
    for path in sorted(trace_dir.glob("*.pt.trace.json.gz")):
        rank = int(RANK_RE.search(path.name).group(1))
        ranks[rank] = steps(load(path))
    result = {"ranks": {}, "collectives": {}}
    for rank, rows in ranks.items():
        phase_busy = collections.defaultdict(list)
        cats = collections.defaultdict(list)
        moe_window = []
        aux = []
        streams = collections.Counter()
        n_steps = len(rows)
        for row in rows:
            for name, ops in row["phases"].items():
                phase_busy[name].append(union(ops))
            target = row["phases"]["target"]
            for name, value in partition(target).items():
                cats[name].append(value)
            for op in target:
                streams[op["args"].get("stream")] += op["end"] - op["t"]
        mains = {s for s, _ in streams.most_common(1)}
        for row in rows:
            target = row["phases"]["target"]
            main = [o for o in target if o["args"].get("stream") in mains]
            ax = [o for o in target if o["args"].get("stream") not in mains]
            # per-layer MoE window: route_prep .. finalize on the main stream
            moe_ops = [o for o in main if "tiered_decode" in o["name"] or "cute_dsl_ll_bf16" in o["name"]
                       or "grouped_topk" in o["name"]]
            moe_window.append(union(moe_ops))
            aux.append((union(ax) if ax else 0.0,
                        intersect(ax, main) if ax else 0.0))
        mean = lambda xs: statistics.fmean(xs) if xs else 0.0  # noqa: E731
        period = [r["period"] for r in rows]
        result["ranks"][rank] = {
            "steps": n_steps,
            "period_us": {"mean": mean(period), "p50": statistics.median(period)},
            "phase_busy_us": {k: mean(v) for k, v in phase_busy.items()},
            "target_categories_us": {k: sum(v) / n_steps for k, v in cats.items()},
            "moe_window_us": mean(moe_window),
            "aux_busy_us": mean(x[0] for x in aux),
            "aux_hidden_us": mean(x[1] for x in aux),
        }
    tables = {rank: collectives(rows) for rank, rows in ranks.items()}
    common = set.intersection(*(set(t) for t in tables.values()))
    by_kind = collections.defaultdict(lambda: {"n": 0, "sum": [], "mins": [], "last": []})
    for key in common:
        values = {rank: tables[rank][key] for rank in tables}
        smallest = min(values.values())
        for rank, v in values.items():
            if v <= smallest * 1.001:
                by_kind[key[1]]["last"].append(rank)
                break
        by_kind[key[1]]["sum"].append(statistics.fmean(values.values()))
        by_kind[key[1]]["mins"].append(smallest)
        by_kind[key[1]]["n"] += 1
    n_steps = min(len(rows) for rows in ranks.values())
    for kind, d in by_kind.items():
        result["collectives"][kind] = {
            "per_step": d["n"] / n_steps,
            "mean_us_per_call": statistics.fmean(d["sum"]),
            "min_us_per_call": statistics.fmean(d["mins"]),
            "waiting_us_per_step": (sum(d["sum"]) - sum(d["mins"])) / n_steps,
            "last_rank": collections.Counter(d["last"]),
        }
    result["n_steps"] = n_steps
    return result


def report(label: str, result: dict) -> None:
    ranks = result["ranks"]
    avg = lambda f: statistics.fmean(f(r) for r in ranks.values())  # noqa: E731

    def fmt_rank(f, digits=2):
        return "{" + ",".join(str(round(f(r), digits)) for r in ranks.values()) + "}"

    ms = lambda us: us / 1000  # noqa: E731
    print(f"\n===== {label}: {result['n_steps']} decode steps, ranks {sorted(ranks)}")
    print(f"step period (GPU start to start): mean {ms(avg(lambda r: r['period_us']['mean'])):.2f} ms "
          f"per rank {fmt_rank(lambda r: ms(r['period_us']['mean']))} "
          f"(p50 {ms(avg(lambda r: r['period_us']['p50'])):.2f})")
    b = lambda k: ms(avg(lambda r: r["phase_busy_us"].get(k, 0)))  # noqa: E731
    print(f"phase busy (ms): target {b('target'):.2f}, logits {b('logits'):.2f}, "
          f"draft+sample {b('draft'):.2f}, host-side {b('host-side'):.2f}")
    print(f"GPU idle inside step: "
          f"{ms(avg(lambda r: r['period_us']['mean'])) - sum(b(k) for k in ('target','logits','draft','host-side')):.2f} ms")

    print(f"\ntarget verify graph, union partition per step (ms, mean over ranks | per rank | share):")
    total = avg(lambda r: sum(r["target_categories_us"].values()))
    rowsk = [(avg(lambda r: r["target_categories_us"].get(c, 0)), c) for c, _ in CATEGORIES
             if any(c in r["target_categories_us"] for r in ranks.values())]
    for value, name in sorted(rowsk, reverse=True):
        v_rank = fmt_rank(lambda r, n=name: ms(r["target_categories_us"].get(n, 0)))
        print(f"  {name:32s} {ms(value):6.2f}  {v_rank}  {value / total * 100:5.1f}%")
    print(f"  {'target busy':32s} {ms(total):6.2f}  (of {ms(avg(lambda r: r['period_us']['mean'])):.2f} ms step)")

    print(f"\nmain-stream decode-MoE window (router..finalize) per step: "
          f"{ms(avg(lambda r: r['moe_window_us'])):.2f} ms "
          f"({avg(lambda r: r['moe_window_us']) / avg(lambda r: r['period_us']['mean']) * 100:.0f}% of step)")
    print(f"aux stream (shared expert) busy {ms(avg(lambda r: r['aux_busy_us'])):.2f} ms/step, "
          f"hidden under main stream {ms(avg(lambda r: r['aux_hidden_us'])):.2f}")

    print(f"\ncross-rank collectives in the target graph (per call us; waiting per step):")
    for kind in sorted(k for k in result["collectives"] if k.split(".")[0] in COLLECTIVE_KEYS):
        d = result["collectives"][kind]
        print(f"  {kind:30s} x{d['per_step']:.0f}  {d['mean_us_per_call']:6.1f} us "
              f"(min-rank floor {d['min_us_per_call']:6.1f}) -> waiting {ms(d['waiting_us_per_step']):5.2f} ms/step "
              f"last={dict(d['last_rank'])}")


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
        args.json.write_text(json.dumps(results, default=str, indent=1))


if __name__ == "__main__":
    main()
