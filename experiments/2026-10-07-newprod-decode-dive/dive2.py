#!/usr/bin/env python3
"""Second-pass dive into the new-prod decode trace. analyze.py's loader gives
ms everywhere; this script stays in ms/µs correctly.

H  all-reduce (fused trtllm) waiting per rank and per step: who arrives last,
   which steps burst, what ran just before the burst window
I  GPU idle decomposition: in-graph gaps (>= threshold) by layer ordinal,
   plus per-phase-gap idle (target end -> draft, -> logits, -> host, -> next)
J  draft region span vs busy (launch starvation), logits/host region spans
K  per-step AR#0 waiting (step entry) and the fused-AR census by position
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402

MS = 1.0
US = 1e-3  # analyze t/end are ms


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


def fmt_us(ms):
    return f"{ms / US:7.2f}"


def union_gaps(ops, thr=0.002):
    """gaps >= thr ms inside the union span of ops -> list of (t, dur)."""
    gaps = []
    cursor = None
    for o in sorted(ops, key=lambda e: e["t"]):
        if cursor is not None and o["t"] - cursor >= thr:
            gaps.append((cursor, o["t"] - cursor))
        cursor = max(cursor, o["end"]) if cursor else o["end"]
    return gaps


def main():
    d = Path(sys.argv[1])
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    res = {}
    ranks = {}
    for path in sorted(d.glob("*.pt.trace.json.gz")):
        rank = int(A.RANK_RE.search(path.name).group(1))
        ranks[rank] = A.steps(A.load(path))
    n = min(len(r) for r in ranks.values())
    for rank in ranks:
        ranks[rank] = ranks[rank][:n]
    r0 = ranks[0]

    # ---- H: fused AR waiting, per rank and per step ----------------------
    ARNAME = "trtllm_allreduce_fusion"
    ar = {}
    for rank, rows in ranks.items():
        table = []
        for index, row in enumerate(rows):
            calls = sorted((o for o in row["phases"]["target"] if ARNAME in o["name"]),
                           key=lambda e: e["t"])
            table.append([(o["t"], o["end"] - o["t"]) for o in calls])
        ar[rank] = table
    n_ar = min(len(s) for r in ar for s in ar[r])
    ar_wait_step = {r: [] for r in ranks}
    ar_wire_step = {r: [] for r in ranks}
    last_arrival = collections.Counter()
    ar0_wait = {r: [] for r in ranks}
    mutual = []
    for i in range(n):
        k = min(len(ar[r][i]) for r in ranks)
        waits = {}
        for r in ranks:
            w = 0.0
            for j in range(k):
                own = ar[r][i][j]
                others_min = min(ar[o][i][j][1] for o in ranks if o != r)
                w += own[1] - others_min
            waits[r] = w
            ar_wire_step[r].append(sum(ar[r][i][j][1] - max(0, ar[r][i][j][1] - min(ar[o][i][j][1] for o in ranks if o != r)) for j in range(k)))
        for r in ranks:
            ar_wait_step[r].append(waits[r])
            ar0_wait[r].append(ar[r][i][0][1] - min(ar[o][i][0][1] for o in ranks if o != r))
        last_arrival[min(waits, key=waits.get)] += 1
        mutual.append(st.fmean([waits[r] for r in ranks]))
    tot_wait = st.fmean([st.fmean(v) for v in ar_wait_step.values()])
    print(f"\nH: fused AR x{n_ar}/step; cross-rank waiting (own minus best-peer, per step)")
    res["H"] = {"mean_wait_ms": tot_wait, "per_rank_wait_ms": {r: st.fmean(v) for r, v in ar_wait_step.items()},
                "first_arrival_counts": {str(r): c for r, c in last_arrival.items()}}
    for r in sorted(ranks):
        v = ar_wait_step[r]
        print(f"   rank {r}: wait {st.fmean(v):6.3f} ms/step  p50 {st.median(v):6.3f}"
              f"  p90 {q(v, .9):6.3f}  max {max(v):6.3f}")
    print(f"   rank with least wait (last to arrive) per step: {dict(last_arrival)}")
    v0 = [st.fmean([ar0_wait[r][i] for r in ranks]) for i in range(n)]
    print(f"   AR#0 (step entry) mean wait {st.fmean(v0):.3f} ms p90 {q(v0, .9):.3f}"
          f" max {max(v0):.3f}; steps >0.5 ms: {sum(x > 0.5 for x in v0)}/{n}")
    res["H"]["ar0_wait"] = {"mean": st.fmean(v0), "p90": q(v0, .9), "max": max(v0),
                            "over_0.5ms": sum(x > 0.5 for x in v0)}

    # burst steps: period far above median; what ran before
    periods = [r["period"] for r in r0]
    med = st.median(periods)
    burst = [i for i, p in enumerate(periods) if p > med + 1.0]
    print(f"\n   burst steps (period > median+1ms): {len(burst)} of {n}: "
          f"{[(i, round(periods[i], 1)) for i in burst]}")
    res["H"]["burst_steps"] = burst
    for i in burst[:4]:
        row = r0[i]
        span = row["span"]
        ph = {k: (min(o["t"] for o in ops), max(o["end"] for o in ops))
              for k, ops in row["phases"].items()}
        print(f"   step {i}: period {periods[i]:.2f}; phases {[f'{k} {v[1]-v[0]:.2f}' for k, v in ph.items()]}")
        # in-graph idle of this step
        gaps = union_gaps([o for o in row["phases"]["target"]], 0.005)
        if gaps:
            print(f"     in-target gaps >5us: {[(round(g[0]-span[0],1), round(g[1]/US,1)) for g in gaps]}")

    # ---- I: idle decomposition ------------------------------------------
    in_graph, phase_gap = collections.Counter(), collections.defaultdict(float)
    gap_sites = collections.Counter()
    for row in r0:
        span = row["span"]
        # count FlashMLA ordinals for gap location
        mla = sorted((o for o in row["phases"]["target"]
                      if "flash_fwd_splitkv_mla" in o["name"]), key=lambda e: e["t"])
        for t, dur in union_gaps(row["phases"]["target"], 0.003):
            in_graph[round(dur / US)] += 1
            if mla:
                import bisect
                ordinal = bisect.bisect([m["t"] for m in mla], t) - 1
                gap_sites[ordinal // 6] += dur
        # phase boundary gaps: target end -> max(others start later)
        tend = max(o["end"] for o in row["phases"]["target"])
        rest = [o for k, ops in row["phases"].items() if k != "target" for o in ops]
        if rest:
            phase_gap["target->first non-target"] += max(0.0, min(o["t"] for o in rest) - tend)
    print(f"\nI: in-target-graph gaps >= 3 us: total {sum(in_graph.values()) * 0} buckets")
    tot_gap_ms = sum(k * c for k, c in in_graph.items()) * US / len(r0)
    print(f"   in-graph gap time: {tot_gap_ms:.3f} ms/step over "
          f"{sum(in_graph.values()) / len(r0):.0f} gaps/step; largest gap buckets (us x count/step): "
          f"{[(k, round(c / len(r0), 2)) for k, c in sorted(in_graph.items(), reverse=True)[:8]]}")
    top_layers = gap_sites.most_common(6)
    print(f"   gap time by 6-layer bucket (top): {[(l, round(v / len(r0), 3)) for l, v in top_layers]} ms/step")
    res["I"] = {"in_graph_gap_ms": tot_gap_ms,
                "buckets": {str(k): c / len(r0) for k, c in sorted(in_graph.items(), reverse=True)[:12]},
                "gap_ms_by_6layer": {str(l): v / len(r0) for l, v in top_layers}}

    # ---- J: draft / logits / host region spans ---------------------------
    reg = collections.defaultdict(list)
    for row in r0:
        for k, ops in row["phases"].items():
            if k != "target" and ops:
                reg[f"{k} span"].append(max(o["end"] for o in ops) - min(o["t"] for o in ops))
                reg[f"{k} busy"].append(A.union(ops))
        all_ops = [o for ops in row["phases"].values() for o in ops]
        reg["all busy"].append(A.union(all_ops))
    print("\nJ: regions (ms/step):")
    for k in ("draft", "logits", "host-side"):
        if f"{k} span" in reg:
            print(f"   {k:10s} span {st.fmean(reg[k + ' span']):.3f} busy {st.fmean(reg[k + ' busy']):.3f}"
                  f" -> gap {(st.fmean(reg[k + ' span']) - st.fmean(reg[k + ' busy'])):.3f}")
    print(f"   total step busy {st.fmean(reg['all busy']):.3f} vs period {st.fmean(periods):.3f}"
          f" -> idle {st.fmean(periods) - st.fmean(reg['all busy']):.3f} ms")
    res["J"] = {k: st.fmean(v) for k, v in reg.items()}

    # ---- K: what the "attention" bucket really is -------------------------
    att = collections.defaultdict(lambda: [0, 0.0])
    for row in r0:
        for o in row["phases"]["target"]:
            if A.category(o) == "attention":
                att[o["name"][:60]][0] += 1
                att[o["name"][:60]][1] += o["end"] - o["t"]
    print("\nK: kernels filed under 'attention' (sum ms/step):")
    for name, (cnt, t) in sorted(att.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"   {t / len(r0):6.3f} ms  x{cnt / len(r0):5.1f}/step  {name}")
        res.setdefault("K", {})[name] = t / len(r0)

    if out:
        Path(out).write_text(json.dumps(res, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
