"""MTP depth A/B over timeline_save.py arms: turns, output tok/s, per-turn
decode rate (output tokens over first token -> end), the same on replay turns
both arms reached, TTFT, and spec-decode acceptance per draft position from
the /metrics snapshots.

    ab_depth.py <arm dir> [<arm dir> ...]
"""
import json
import re
import statistics as st
import sys
from pathlib import Path


def spec(path):
    c = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r'^vllm:spec_decode_(num_drafts|num_accepted_tokens|num_draft_tokens|'
                     r'num_accepted_tokens_per_pos)_total\{([^}]*)\} (\S+)', line)
        if m:
            p = re.search(r'position="(\d+)"', m.group(2))
            k = m.group(1) + (f"@{p.group(1)}" if p else "")
            c[k] = c.get(k, 0) + float(m.group(3))
    return c


arms = {}
for d in map(Path, sys.argv[1:]):
    ev = [json.loads(l) for l in (d / "events.jsonl").open()]
    ok = [e for e in ev if not e["err"] and e["first"] is not None and (e["output"] or 0) > 20]
    span = max(e["end"] for e in ev)
    out = sum(e["output"] or 0 for e in ev if not e["err"])
    rate = {e["index"]: (e["output"] - 1) / (e["end"] - e["first"]) for e in ok
            if e["end"] > e["first"]}
    s0, s1 = spec(d / "metrics-start.txt"), spec(d / "metrics-end.txt")
    dr = s1["num_drafts"] - s0["num_drafts"]
    pos = [(s1[k] - s0.get(k, 0)) / dr for k in sorted(s1) if k.startswith("num_accepted_tokens_per_pos@")]
    ttft = [e["first"] - e["send"] for e in ev if not e["err"] and e["first"]]
    arms[d.name] = rate
    print(f"{d.name}: turns {len(ev)}, out {out} tok, {out / span:.1f} tok/s over {span:.0f} s")
    print(f"  per-turn decode tok/s mean {st.mean(rate.values()):.1f} median "
          f"{st.median(rate.values()):.1f} (n {len(rate)}); TTFT mean {st.mean(ttft):.2f} "
          f"median {st.median(ttft):.2f}")
    print(f"  drafts {dr:.0f}, accept per position " + " ".join(f"{p:.3f}" for p in pos)
          + f", tokens/round {1 + sum(pos):.2f}")
names = list(arms)
for a in names[1:]:
    both = sorted(set(arms[names[0]]) & set(arms[a]))
    if both:
        r = [arms[a][i] / arms[names[0]][i] for i in both]
        print(f"{a} / {names[0]} decode rate on {len(both)} shared turns: median {st.median(r):.3f}, "
              f"geomean {st.geometric_mean(r):.3f}")
