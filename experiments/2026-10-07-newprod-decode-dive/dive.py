#!/usr/bin/env python3
"""Kernel-level dive into the new-prod decode trace (skipkv-newprod, 2026-10-07).

Section A  step periods, burst steps, span decomposition
Section B  target-graph kernel census (count/step, mean us, solo vs shared)
Section C  all-reduces (incl. fused AR+RMS) and one-shots: cross-rank aligned
           wire time vs waiting, per step
Section D  attention chain: FlashMLA anchor vs skip layers, combines, indexer
Section E  between-target region: draft / logits / host-side + step-boundary gaps
Section F  side-stream skip-KV staging: span, slack, exposure
Section G  per-step cross-rank period spread and laggard identity

Usage: dive.py <trace-window-dir> [--json out.json]
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402

US = 1000.0


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


def census(rows, phase="target"):
    """name -> (count, sum, solo-sum) over all steps of one rank."""
    tot = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for row in rows:
        ops = row["phases"][phase]
        covered = []
        for o in sorted(ops, key=lambda e: e["t"]):
            overlap = sum(min(o["end"], e["end"]) - max(o["t"], e["t"])
                          for e in covered if e["end"] > o["t"])
            row_tot = o["end"] - o["t"]
            tot[o["name"]][0] += 1
            tot[o["name"]][1] += row_tot
            tot[o["name"]][2] += row_tot - overlap
            covered.append(o)
    return tot


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

    # --- A: periods / burst / span decomposition ------------------------
    r0 = ranks[0]
    periods = [r["period"] for r in r0]
    spans = [r["span"] for r in r0]
    busy = {ph: [A.union(r["phases"][ph]) for r in r0]
            for ph in ("target", "logits", "draft", "host-side")}
    res["A"] = {
        "steps": n,
        "period_ms": {"mean": st.fmean(periods), "p50": st.median(periods),
                      "p90": q(periods, .9), "p99": q(periods, .99), "max": max(periods)},
        "busy_ms_means": {k: st.fmean(v) for k, v in busy.items()},
        "target_span_ms": st.fmean(s[1] - s[0] for s in spans),
    }
    print(f"A: {n} steps; period mean {st.fmean(periods):.2f} p50 {st.median(periods):.2f} "
          f"p90 {q(periods, .9):.2f} p99 {q(periods, .99):.2f} max {max(periods):.2f} ms")
    print(f"   target span {res['A']['target_span_ms']:.2f} ms; busy: "
          + ", ".join(f"{k} {st.fmean(v):.3f}" for k, v in busy.items()))
    excess = [p - (s[1] - s[0]) for p, s in zip(periods, spans)]
    print(f"   period - target span: mean {st.fmean(excess):.2f} p90 {q(excess, .9):.2f} ms")

    # --- B: kernel census of the target graph (rank 0) -------------------
    tot = census(r0)
    n_t = sum(v[0] for v in tot.values()) / len(r0)
    res["B"] = {"kernels_per_step": n_t, "top": []}
    print(f"\nB: {n_t:.0f} target kernels/step; top by shared time (us/step):")
    for name, (cnt, tot_t, solo) in sorted(tot.items(), key=lambda kv: -kv[1][1])[:45]:
        cps = cnt / len(r0)
        res["B"]["top"].append({"name": name[:120], "count_per_step": cps,
                                "mean_us": tot_t / cnt if cnt else 0,
                                "shared_us_per_step": tot_t / len(r0),
                                "solo_us_per_step": solo / len(r0)})
        print(f"   {tot_t / len(r0):8.1f} us  x{cps:6.1f}  mean {tot_t / cnt:6.2f}  {name[:95]}")

    # --- C: collectives across ranks (fused AR + one-shots) --------------
    # find names that could be the fused allreduce+add+rmsnorm kernel
    cand = {name for name in tot if any(s in name.lower() for s in
            ("allreduce", "all_reduce", "ar_fusion", "fused_all", "cross_device"))}
    oneshot = {name for name in tot if "one_shot::" in name}
    print(f"\nC: collective kernel name candidates: {sorted(cand)[:8]}")
    tables = {}
    for rank, rows in ranks.items():
        table = collections.defaultdict(dict)
        for index, row in enumerate(rows):
            counts = collections.Counter()
            for o in sorted(row["phases"]["target"], key=lambda e: e["t"]):
                key = None
                if any(c in o["name"] for c in cand):
                    key = "AR"
                elif "one_shot::" in o["name"]:
                    key = o["name"][o["name"].index("one_shot::"):][:28]
                if key:
                    table[key][(index, counts[key])] = o["end"] - o["t"]
                    counts[key if key == "AR" else key] += 1
        tables[rank] = table
    res["C"] = {}
    keys = sorted(set(tables[0]))
    for key in keys:
        common = set.intersection(*(set(tables[r][key]) for r in tables))
        wire, mean, cnts = [], [], []
        by_step = collections.defaultdict(list)
        for (index, ordinal) in common:
            vals = [tables[r][key][(index, ordinal)] for r in tables]
            wire.append(min(vals))
            mean.append(st.fmean(vals))
            by_step[index].append(st.fmean(vals) - min(vals))
        per_step = {k: sum(v) for k, v in by_step.items()}
        row = {"count_per_step": len(common) / n,
               "mean_us": st.fmean(mean), "wire_us": st.fmean(wire),
               "wait_us_per_step": st.fmean(per_step.values()),
               "wait_p90_step_ms": q(list(per_step.values()), .9) / US,
               "wait_max_step_ms": max(per_step.values()) / US}
        res["C"][key] = row
        print(f"   {key[:38]:38s} x{row['count_per_step']:5.1f}  mean {row['mean_us']:6.2f}"
              f"  wire {row['wire_us']:6.2f}  -> wait {row['wait_us_per_step']:7.1f} us/step"
              f"  (p90 {row['wait_p90_step_ms']:.2f} max {row['wait_max_step_ms']:.2f} ms/step)")
    worst = sorted(per_step.items(), key=lambda kv: -kv[1])[:8] if per_step else []
    print(f"   worst AR-wait steps (us): {[(k, round(v,1)) for k, v in worst]}")

    # --- D: attention chain ------------------------------------------------
    import re
    MLA = re.compile(r"flash_fwd_splitkv_mla_fp8_sparse")
    ANCHORS = sorted({0, 1, 2} | set(range(6, 78, 4)))
    dsa = {name: v for name, v in tot.items()
           if A.category({"name": name}) == "DSA indexer"}
    at = collections.defaultdict(list)  # per-step aggregate per kernel family
    fam = {"mla_anchor": [], "mla_skip": [], "mla_other": []}
    for row in r0:
        mla = [o for o in row["phases"]["target"] if MLA.search(o["name"])]
        mla.sort(key=lambda e: e["t"])
        for i, o in enumerate(mla):
            if i < 78:
                (fam["mla_anchor"] if i in ANCHORS else fam["mla_skip"]).append(o["end"] - o["t"])
            else:
                fam["mla_other"].append(o["end"] - o["t"])
    res["D"] = {}
    for k, v in fam.items():
        if v:
            res["D"][k] = {"n": len(v) / len(r0), "mean_us": st.fmean(v), "p50_us": st.median(v),
                           "p90_us": q(v, .9)}
            print(f"\nD: {k}: {len(v) / len(r0):.1f}/step, mean {st.fmean(v):.2f}"
                  f" p50 {st.median(v):.2f} p90 {q(v, .9):.2f} us"
                  f" -> {st.fmean(v) * len(v) / len(r0) / US:.2f} ms/step")
    for name, (cnt, t_sum, solo) in sorted(dsa.items(), key=lambda kv: -kv[1][1])[:6]:
        res["D"][f"dsa:{name[:60]}"] = {"count_ps": cnt / len(r0), "mean_us": t_sum / cnt}
        print(f"   DSA {cnt / len(r0):5.1f} x {t_sum / cnt:6.2f} us  {name[:80]}")

    # --- E: between-target region ------------------------------------------
    # per step: everything after target span end -> next step start
    gaps = []
    draught_busy = [A.union(r["phases"]["draft"]) for r in r0]
    host_busy = [A.union(r["phases"]["host-side"]) for r in r0]
    res["E"] = {"draft_busy_ms": st.fmean(draught_busy), "host_side_busy_ms": st.fmean(host_busy)}
    print(f"\nE: draft busy {st.fmean(draught_busy):.3f} ms, host-side busy "
          f"{st.fmean(host_busy):.3f} ms, logits busy above")
    draft_census = census(r0, "draft")
    res["E"]["draft_top"] = []
    print("   draft kernels top (us/step):")
    for name, (cnt, t_sum, solo) in sorted(draft_census.items(), key=lambda kv: -kv[1][1])[:12]:
        res["E"]["draft_top"].append({"name": name[:100], "count_ps": cnt / len(r0),
                                      "mean_us": t_sum / cnt if cnt else 0,
                                      "sum_us": t_sum / len(r0)})
        print(f"   {t_sum / len(r0):8.1f} us  x{cnt / len(r0):6.1f}  mean {t_sum / cnt:6.2f}  {name[:80]}")

    # --- F: skip-KV staging kernels ----------------------------------------
    side = [o for row in r0 for o in row["phases"]["target"]
            if any(k in o["name"] for k in ("_gather_rows_kernel", "_claim_rows", "_remap_rows",
                                             "skip_kv", "reserved"))]
    if side:
        res["F"] = {"count_ps": len(side) / len(r0),
                    "mean_us": st.fmean(o["end"] - o["t"] for o in side),
                    "sum_us_ps": sum(o["end"] - o["t"] for o in side) / len(r0)}
        print(f"\nF: staging kernels {len(side) / len(r0):.1f}/step, mean "
              f"{st.fmean(o['end'] - o['t'] for o in side):.2f} us, sum "
              f"{sum(o['end'] - o['t'] for o in side) / len(r0):.1f} us/step")
    else:
        print("\nF: no staging kernels found in target phase")

    # --- G: cross-rank period spread and laggard ---------------------------
    spread, firsts = [], collections.Counter()
    for i in range(n):
        ps = {r: ranks[r][i]["period"] for r in ranks}
        spread.append(max(ps.values()) - min(ps.values()))
        firsts[max(ps, key=ps.get)] += 1
    res["G"] = {"spread_p50_us": st.median(spread), "spread_p90_us": q(spread, .9),
                "laggard_counts": dict(firsts)}
    print(f"\nG: cross-rank period spread p50 {st.median(spread):.1f} p90 {q(spread, .9):.1f}"
          f" us; laggard {dict(firsts)}")
    if out:
        Path(out).write_text(json.dumps(res, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
