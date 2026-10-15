"""Compare timeline_run.py arms (FCFS vs SRPT): turns done, TTFT stats (all,
and turns past each agent's first), output tok/s, prefix hit rate.

    ab_srpt.py <arm dir> [<arm dir> ...]
"""
import json
import statistics
import sys
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


for d in map(Path, sys.argv[1:]):
    ev = [json.loads(l) for l in (d / "events.jsonl").open()]
    ok = [e for e in ev if not e["err"] and e["first"] is not None]
    ttft = [e["first"] - e["send"] for e in ok]
    follow = [e["first"] - e["send"] for e in ok if e["k"] > 0]
    sm = [json.loads(l) for l in (d / "samples.jsonl").open()]
    sm = [s for s in sm if "prefix_cache_queries_total" in s]
    q = sm[-1]["prefix_cache_queries_total"] - sm[0]["prefix_cache_queries_total"]
    h = sm[-1]["prefix_cache_hits_total"] - sm[0]["prefix_cache_hits_total"]
    span = max(e["end"] for e in ev)
    out = sum(e["output"] or 0 for e in ok)
    print(f"{d.name}: turns {len(ok)} (err {len(ev) - len(ok)}), out {out} tok "
          f"({out / span:.1f} tok/s over {span:.0f} s), prefix hit {100 * h / q:.1f}%")
    print(f"  TTFT all     mean {statistics.mean(ttft):6.2f}  p50 {pct(ttft, 50):6.2f}  "
          f"p90 {pct(ttft, 90):6.2f}  max {max(ttft):6.1f}  sum {sum(ttft):7.1f}")
    print(f"  TTFT follow  mean {statistics.mean(follow):6.2f}  p50 {pct(follow, 50):6.2f}  "
          f"p90 {pct(follow, 90):6.2f}  max {max(follow):6.1f}  n {len(follow)}")
    print("  slow turns (>10 s): " + ", ".join(
        f"#{e['index']}:{e['first'] - e['send']:.0f}s/{e['input'] // 1000}K"
        for e in ok if e["first"] - e["send"] > 10))
