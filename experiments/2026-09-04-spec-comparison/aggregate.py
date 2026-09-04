#!/usr/bin/env python3
"""Aggregate the four comparison arms into one table.

Standalone: takes only the *-result.json and *-server.out files, so results
can be processed from a cold start.

Leads with acceptance length and step time. Those are robust to what the
arms actually generated; at temperature 1.0 each arm writes different text,
so tokens/s is secondary and reported per decode second, never per wall
second -- wall includes ~7s of prefill per request and dividing by it
understates decode by more than the effect under study.
"""

import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
ARMS = [("spdf23", "DFlash2 K=3"), ("spdf27", "DFlash2 K=7"),
        ("spmtp3", "MTP K=3"), ("spmtp7", "MTP K=7")]


def residency(label: str) -> str:
    f = HERE / f"{label}-server.out"
    if not f.exists():
        return "-"
    m = re.search(r"(\d+) hot / (\d+) cold experts per rank",
                  f.read_text(errors="replace"))
    return f"{m.group(1)}h/{m.group(2)}c" if m else "-"


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main() -> int:
    arms = {}
    for label, name in ARMS:
        f = HERE / f"{label}-result.json"
        if not f.exists():
            print(f"  (missing {f.name})", file=sys.stderr)
            continue
        arms[label] = (name, json.load(open(f)))
    if not arms:
        print("no results yet")
        return 1

    print("## Correctness: temperature-0 probes must match across arms\n")
    ref_label = next(iter(arms))
    ref = arms[ref_label][1]["probes"]
    for i, p in enumerate(ref):
        vals = {lbl: d["probes"][i]["content"].strip() for lbl, (_, d) in arms.items()}
        uniq = set(vals.values())
        blank = all(v == "" for v in vals.values())
        if blank:
            verdict = "VACUOUS (all empty: reasoning consumed the cap; proves nothing)"
        elif len(uniq) == 1:
            verdict = f"match: {next(iter(uniq))[:40]!r}"
        else:
            verdict = f"MISMATCH {vals}"
        print(f"  probe {i}: {verdict}")

    print("\n## Headline (acceptance and step time lead; both robust to output variation)\n")
    hdr = (f"{'arm':<12} {'AL':>6} {'step_ms':>8} {'tok/s_dec':>10} "
           f"{'ttft_med':>9} {'out_tok':>8} {'experts':>12} {'think_only':>11}")
    print(hdr); print("-" * len(hdr))
    summary = {}
    for label, (name, d) in arms.items():
        rows = d["per_request"]
        # Drop request 1: three Triton kernels JIT-compile on first inference,
        # so its TTFT and step time carry compile cost on every arm.
        warm = rows[1:] if len(rows) > 1 else rows
        steps = sum(r["steps"] for r in warm)
        acc = sum(r["accepted"] for r in warm)
        al = (acc + steps) / steps if steps else float("nan")
        dec_s = sum(r["decode_s"] for r in warm)
        out = sum(r["output_tokens"] or 0 for r in warm)
        # spdf23's bench started before think_closed was recorded; for it the
        # count is unknown rather than zero, and is reported as such.
        has_flag = all("think_closed" in r for r in warm)
        think_only = (sum(1 for r in warm if not r["think_closed"])
                      if has_flag else None)
        summary[label] = {
            "name": name, "al": al,
            "step_ms": med([r.get("step_ms_median") for r in warm]),
            "tok_s_decode": out / dec_s if dec_s else None,
            "ttft_med": med([r["ttft_s"] for r in warm]),
            "out": out, "n": len(warm), "think_only": think_only,
        }
        s = summary[label]
        tt = "  n/a" if think_only is None else f"{think_only:>4}"
        print(f"{name:<12} {al:>6.3f} {s['step_ms'] or 0:>8.1f} "
              f"{s['tok_s_decode'] or 0:>10.1f} {s['ttft_med'] or 0:>9.2f} "
              f"{out:>8} {residency(label):>12} {tt}/{len(warm):<6}")

    print("\n## By task kind (acceptance)\n")
    print(f"{'arm':<12} {'code AL':>9} {'prose AL':>9} {'code tok/s':>11} {'prose tok/s':>12}")
    for label, (name, d) in arms.items():
        cells = []
        for kind in ("code", "prose"):
            rs = [r for r in d["per_request"][1:] if r["task_kind"] == kind]
            st = sum(r["steps"] for r in rs); ac = sum(r["accepted"] for r in rs)
            ds = sum(r["decode_s"] for r in rs); ot = sum(r["output_tokens"] or 0 for r in rs)
            cells.append(((ac + st) / st if st else 0, ot / ds if ds else 0))
        print(f"{name:<12} {cells[0][0]:>9.3f} {cells[1][0]:>9.3f} "
              f"{cells[0][1]:>11.1f} {cells[1][1]:>12.1f}")

    print("\n## Per-position acceptance (share of steps reaching each draft slot)\n")
    for label, (name, d) in arms.items():
        tot = {}
        for r in d["per_request"][1:]:
            for k, v in r["per_pos"].items():
                tot[int(k)] = tot.get(int(k), 0) + v
        steps = sum(r["steps"] for r in d["per_request"][1:])
        cells = " ".join(f"{tot.get(i,0)/steps*100:5.1f}"
                         for i in range(max(tot) + 1)) if tot else "-"
        print(f"{name:<12} pos0..N: {cells}")

    print("\n## Sample answers (post-</think>), one code and one prose per arm\n")
    for label, (name, d) in arms.items():
        for kind in ("code", "prose"):
            r = next((x for x in d["per_request"][1:]
                      if x["task_kind"] == kind and x.get("answer_sample")), None)
            if r:
                txt = " ".join(r["answer_sample"].split())[:150]
                print(f"  {name:<12} {kind:<6} {txt!r}")
            else:
                print(f"  {name:<12} {kind:<6} (no sample recorded)")

    if len(summary) == 4:
        print("\n## Verdict\n")
        best = max(summary.values(), key=lambda s: s["tok_s_decode"] or 0)
        print(f"  fastest decode: {best['name']} at {best['tok_s_decode']:.1f} tok/s")
        for s in summary.values():
            if s["think_only"] is not None and s["think_only"] > s["n"] * 0.5:
                print(f"  WARNING {s['name']}: {s['think_only']}/{s['n']} requests "
                      f"never finished reasoning -- this is a reasoning-throughput "
                      f"comparison for that arm, not code-vs-prose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
