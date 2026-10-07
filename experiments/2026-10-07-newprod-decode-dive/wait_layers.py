#!/usr/bin/env python3
"""Is the fused-AR wait concentrated in specific layers, and does it track
per-layer MoE skew? Per (step, layer ordinal 0..77): w13 duration per rank ->
max-minus-mean skew; AR-wait per ordinal (own minus best peer). Correlating the
two decides 'balanceable via placement' vs 'order statistic'.

Usage: wait_layers.py <trace-window-dir> [--compare]"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-decode-profile"))
import analyze as A  # noqa: E402

US = 1e-3


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


def load_ranks(d):
    ranks = {}
    for path in sorted(d.glob("*.pt.trace.json.gz")):
        rank = int(A.RANK_RE.search(path.name).group(1))
        ranks[rank] = A.steps(A.load(path))
    n = min(len(r) for r in ranks.values())
    return {r: rows[:n] for r, rows in ranks.items()}, n


def w13_table(rows):
    """(step, ordinal) -> duration for the one-kernel w13 gemm."""
    table = {}
    for i, row in enumerate(rows):
        calls = sorted((o for o in row["phases"]["target"]
                        if "tiered_decode::gemm_kernel<0" in o["name"]),
                       key=lambda e: e["t"])
        for j, o in enumerate(calls):
            table[(i, j)] = o["end"] - o["t"]
    return table


def ar_table(rows):
    table = {}
    for i, row in enumerate(rows):
        calls = sorted((o for o in row["phases"]["target"]
                        if "trtllm_allreduce_fusion" in o["name"]), key=lambda e: e["t"])
        for j, o in enumerate(calls):
            table[(i, j)] = o["end"] - o["t"]
    return table


def main():
    d = Path(sys.argv[1])
    ranks, n = load_ranks(d)
    r0 = ranks[0]
    w13 = {r: w13_table(rows) for r, rows in ranks.items()}
    ar = {r: ar_table(rows) for r, rows in ranks.items()}
    common_w = set(w13[0])
    common_a = set(ar[0])
    for r in ranks:
        common_w &= set(w13[r])
        common_a &= set(ar[r])
    n_w = max(j for _, j in common_w) + 1

    # per (step, layer): skew = max-min across ranks of w13 duration
    skew = collections.defaultdict(list)   # layer -> [skew us]
    wmax = collections.defaultdict(list)   # layer -> [max rank duration]
    per_step_skew = collections.defaultdict(float)
    per_step_wait = collections.defaultdict(float)
    mla_skew_all = []
    for i in range(n):
        for j in range(0, n_w, 1):
            key = (i, j)
            if key not in common_w:
                continue
            vals = [w13[r][key] for r in ranks]
            skew[j].append((max(vals) - min(vals)) / US)
            wmax[j].append(st.fmean(vals) / US)
            per_step_skew[i] += (max(vals) - st.fmean(vals)) / US
    # per (step, ordinal) AR wait
    wait_by_ord = collections.defaultdict(list)
    last_by_ord = collections.Counter()
    for i in range(n):
        k = max(j for _, j in common_a if j // 3 < 10**9) if common_a else 0
    k = max(j for _, j in common_a)
    for i in range(n):
        for j in range(k + 1):
            key = (i, j)
            if key not in common_a:
                continue
            vals = {r: ar[r][key] for r in ranks}
            w = max(vals.values()) - min(vals.values())
            wait_by_ord[j].append(w / US)
            per_step_wait[i] += w / US
            last_by_ord[(j, min(vals, key=vals.get))] += 1

    pairs = [(per_step_skew[i], per_step_wait[i]) for i in sorted(per_step_skew)]
    xs, ys = zip(*pairs)
    mx, my = st.fmean(xs), st.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs) / len(pairs)
    r = cov / (st.pstdev(xs) * st.pstdev(ys))
    print(f"w13 layers/step {n_w}; fused-AR calls/step {k + 1}")
    print(f"per-step w13 skew (max-mean summed over layers): mean {mx:.1f} us")
    print(f"per-step fused-AR wait (max-min summed over calls): mean {my:.1f} us")
    print(f"corr(skew, wait) = {r:.3f}  (n={len(pairs)} steps)")

    print("\ntop 10 worst layers by w13 cross-rank skew (mean us, share of step skew):")
    lay = sorted(((j, st.fmean(v), st.fmean(wmax[j])) for j, v in skew.items()),
                 key=lambda t: -t[1])
    tot = sum(v[1] for v in lay)
    for j, s, m in lay[:10]:
        print(f"   layer {j:3d}: skew {s:6.1f} us ({s / tot * 100:4.1f}%)  mean dur {m:6.1f} us")
    print("   (uniform share for 75 layers would be 1.33% each)")

    print("\nAR wait by ordinal position: worst 12 of", k + 1)
    ords = sorted(((j, st.fmean(v)) for j, v in wait_by_ord.items()), key=lambda t: -t[1])
    for j, s in ords[:12]:
        late = collections.Counter()
        for (jj, rr), c in last_by_ord.items():
            if jj == j:
                late[rr] += c
        print(f"   AR #{j:3d}: wait {s:6.1f} us  last-arrive by rank {dict(late)}")
    wait_ms = st.fmean([per_step_wait[i] for i in per_step_wait]) / 1000
    print(f"\ntotal fused-AR wait per step: {wait_ms:.3f} ms")


if __name__ == "__main__":
    main()
