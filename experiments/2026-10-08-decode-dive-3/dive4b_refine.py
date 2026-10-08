#!/usr/bin/env python3
"""Refined numbers for the dd3-w768 dive: decode_gemm classes by position in
the layer window, attn-AR vs MoE-AR split with rank0 wait over the cross-rank
minimum, dense layers 0-2, skip-staging slack, PDL overlap on the main chain.

    dive4b_refine.py <trace-window-dir>
"""
import collections
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402
import gzip
import json as _json


def _load_us(path):
    with gzip.open(path, "rt") as handle:
        events = [e for e in _json.load(handle)["traceEvents"] if e.get("ph") == "X"]
    for event in events:
        event["t"] = event["ts"]
        event["end"] = event["t"] + event.get("dur", 0)
    return events


A.load = _load_us
US = 1e3


def windows_of(step):
    flash = sorted((o for o in step["phases"]["target"]
                    if "flash_fwd_splitkv_mla_fp8_sparse" in o["name"]), key=lambda e: e["t"])
    bounds = [f["t"] for f in flash] + [flash[-1]["end"] + 300.0]
    return flash, bounds


def refine_decode_gemm(rows):
    classes = collections.defaultdict(list)      # class -> per-call us
    perwin = collections.Counter()               # calls per layer window
    anchors = {0, 1, 2} | {k for k in range(6, 78, 4)}
    for step in rows:
        flash, bounds = windows_of(step)
        for k in range(len(flash)):
            ws, we = bounds[k], bounds[k + 1]
            calls = sorted((o for o in step["phases"]["target"]
                            if "decode_gemm_kernel" in o["name"] and ws <= o["t"] < we),
                           key=lambda e: e["t"])
            perwin[len(calls)] += 1
            if len(calls) == 3:
                for idx, name in enumerate(("o_proj", "qkv_a", "q_b")):
                    classes[name].append(calls[idx]["end"] - calls[idx]["t"])
            elif len(calls) == 4 and k in anchors:
                # wq_b runs just before the anchor's FlashMLA? check order:
                # calls[0] is this window's o_proj only when > flash; here
                # detect wq_b as the call preceding the mqa_logits of this layer
                mq = [o for o in step["phases"]["target"] if "mqa_logits" in o["name"]
                      and ws <= o["t"] < we]
                if mq:
                    wq = [c for c in calls if c["t"] < mq[0]["t"]]
                    idx0 = len(wq)  # calls after mqa start with o_proj
                    if len(calls) - idx0 == 3:
                        classes["wq_b (indexer)"].append(wq[-1]["end"] - wq[-1]["t"])
                        for j, name in enumerate(("o_proj", "qkv_a", "q_b")):
                            classes[name].append(calls[idx0 + j]["end"] - calls[idx0 + j]["t"])
    print("--- decode_gemm by position in the layer window")
    print(f"    window call-count histogram: {dict(perwin)}")
    bytes_mib = {"o_proj": 48, "qkv_a": 31, "q_b": 16, "wq_b (indexer)": 16}
    for name, vals in classes.items():
        m = st.mean(vals)
        mb_us = bytes_mib[name] * 1.0486 / m
        print(f"    {name:16} n={len(vals) / len(rows):5.1f}/step  mean {m:6.2f} us  p10 {sorted(vals)[len(vals)//10]:5.1f}"
              f"  {mb_us * 1000:5.0f} GB/s  {mb_us / 3.64 * 100:3.0f}% HBM floor")


def refine_ar(rows, all_ranks):
    # (kind, global ordinal) -> per-rank durations; kind by position in window
    tables = collections.defaultdict(dict)
    for rank, rws in all_ranks.items():
        counter = collections.Counter()
        for step in rws:
            for wk in ("attn", "moe"):
                pass
            # classify per layer: attn AR = the AR between combine and topk;
            # MoE AR = AR after add_mul
            flash, bounds = windows_of(step)
            for k in range(len(flash)):
                ws, we = bounds[k], bounds[k + 1]
                ars = sorted((o for o in step["phases"]["target"]
                              if "allreduce_fusion" in o["name"] and ws <= o["t"] < we),
                             key=lambda e: e["t"])
                for kind, o in zip(("attn", "moe"), ars[:2]):
                    tables[(kind, (counter["step"], k))][rank] = o["end"] - o["t"]
            counter["step"] += 1
    print("--- fused AR by position (rank0; wire = cross-rank min per call)")
    for kind in ("attn", "moe"):
        cells = [per for (k2, _), per in tables.items() if k2 == kind and len(per) == len(all_ranks)]
        if not cells:
            continue
        r0 = [per[0] for per in cells]
        wire = [min(per.values()) for per in cells]
        wait = [a - b for a, b in zip(r0, wire)]
        print(f"    {kind}-AR {len(cells) / len(rows):5.1f}/step  rank0 mean {st.mean(r0):5.1f} us"
              f"  wire(min) {st.mean(wire):5.1f}  rank0 wait {st.mean(wait):5.1f}"
              f"  p90 wait {sorted(wait)[int(len(wait)*.9)]:6.1f}"
              f"  -> busy {st.mean(r0) * len(cells) / len(rows) / US:5.2f} ms/step,"
              f" wait {st.mean(wait) * len(cells) / len(rows) / US:5.2f} ms/step")
    # which rank waits least overall
    for rank in sorted(all_ranks):
        cells = [per for (k2, _), per in tables.items() if len(per) == len(all_ranks)]
        own = st.mean(per[rank] for per in cells)
        wire = st.mean(min(per.values()) for per in cells)
        print(f"      rank{rank}: mean AR {own:5.1f} vs wire {wire:5.1f} -> excess {own - wire:5.1f} us/call")


def dense_layers(rows):
    step = min(rows, key=lambda r: abs(st.median(r2["period"] for r2 in rows) - r["period"]))
    flash, bounds = windows_of(step)
    for k in (0, 1, 2):
        ws, we = bounds[k], bounds[k + 1]
        calls = [o for o in step["phases"]["target"] if o["end"] > ws and o["t"] < we]
        calls.sort(key=lambda e: e["t"])
        print(f"\n--- layer {k} (dense MLP) {we - ws:.1f} us")
        prev = ws
        for o in calls:
            ov = "+" if o["t"] < prev - 0.3 else " "
            prev = max(prev, o["end"])
            print(f"  {o['t'] - ws:7.1f} {o['end'] - o['t']:6.1f} {ov}  {o['name'].split('<')[0].split('(')[0].replace('void ', '')[:70]}")


def staging_slack(rows):
    slacks = []
    for step in rows:
        flash, bounds = windows_of(step)
        for k in range(len(flash) - 1):
            ws, we = bounds[k], bounds[k + 1]
            g = [o for o in step["phases"]["target"] if "_gather_rows" in o["name"]
                 and ws <= o["t"] < we]
            if g:
                slacks.append(bounds[k + 1] - max(o["end"] for o in g))
    if slacks:
        print(f"--- skip-KV staging: gather end -> first skip FlashMLA slack (us): "
              f"min {min(slacks):.1f} p10 {sorted(slacks)[len(slacks)//10]:.1f} n={len(slacks)}")


def pdl_overlap(rows):
    span_us, dur_sum, hidden_act, hidden_fin = [], [], [], []
    for step in rows:
        flash, bounds = windows_of(step)
        for k in range(len(flash)):
            ws, we = bounds[k], bounds[k + 1]
            ops = sorted((o for o in step["phases"]["target"]
                          if ws <= o["t"] < we and "tiered_decode" in o["name"]),
                         key=lambda e: e["t"])
            if len(ops) == 5:
                rp, w13, act, w2, fin = ops
                span_us.append(fin["end"] - rp["t"])
                dur_sum.append(sum(o["end"] - o["t"] for o in ops))
                hidden_act.append(max(0.0, min(w13["end"], act["end"]) - act["t"]))
                hidden_fin.append(max(0.0, min(w2["end"], fin["end"]) - fin["t"]))
    if span_us:
        print(f"--- PDL on the tiered chain (us/layer, n={len(span_us)} 5-kernel windows):")
        print(f"    span {st.mean(span_us):6.1f}   sum of 5 kernel durations {st.mean(dur_sum):6.1f}"
              f"   -> overlapped {st.mean(dur_sum) - st.mean(span_us):6.1f}")
        print(f"    act inside w13 {st.mean(hidden_act):5.1f} of its duration;"
              f" finalize inside w2 {st.mean(hidden_fin):5.1f}")


def main():
    d = Path(sys.argv[1])
    ranks = {}
    for path in sorted(d.glob("*.pt.trace.json.gz")):
        rank = int(A.RANK_RE.search(path.name).group(1))
        ranks[rank] = A.steps(A.load(path))
    n = min(len(r) for r in ranks.values())
    ranks = {r: rows[:n] for r, rows in ranks.items()}
    rows = ranks[0]
    print(f"== {d.name}: {len(rows)} steps")
    refine_decode_gemm(rows)
    refine_ar(rows, ranks)
    staging_slack(rows)
    pdl_overlap(rows)
    dense_layers(rows)


if __name__ == "__main__":
    main()
