"""Kernel-to-kernel edges in the verify graph's 4-layer cycle: for each
consecutive pair in start order, gap = successor start - end of the kernel
before it (negative = it started before its predecessor ended, i.e. a PDL
early launch or a concurrent branch). Medians over all layers of a kind.

    edges_nsys.py report.sqlite [--device ...]
"""
import argparse
import collections
import statistics as st
import sys

sys.argv, argv = sys.argv[:1] + ["--json", "/dev/null"] + sys.argv[1:], sys.argv
ns = {}
src = open(__file__.replace("edges_nsys.py", "cycle_nsys.py")).read()
src = src.split("# per layer kind")[0]
exec(compile(src, "cycle_nsys.py", "exec"), ns)
graphs, layers = ns["graphs"], ns["layers"]
edges = collections.defaultdict(lambda: collections.defaultdict(list))
first = {}
for ks in graphs:
    L = layers(ks)
    pos, kinds = 1, ["prologue"]
    for k, _ in L[1:]:
        pos = 0 if k.startswith("A") else pos + 1
        kinds.append(k if k.endswith("dense") else ("A" if pos == 0 else f"S{pos}"))
    for li, ((_, seg), kind) in enumerate(zip(L, kinds)):
        if kind not in ("A", "S2") or li + 1 >= len(L):
            continue
        nxt = L[li + 1][1][0]
        seq = list(seg) + [nxt]
        occ = collections.Counter()
        for j in range(1, len(seq)):
            p, q = seq[j - 1], seq[j]
            occ[(p[2], q[2])] += 1
            key = (p[2], q[2], occ[(p[2], q[2])])
            edges[kind][key].append(((q[0] - p[1]) / 1e3, (q[0] - p[0]) / 1e3, (p[1] - p[0]) / 1e3))
            first.setdefault((kind, key), (li, j))
for kind in ("S2", "A"):
    print(f"\n== {kind}: predecessor -> successor (start order), median over {len(graphs)} graphs x layers")
    print(f"  {'gap':>6} {'p10':>6} {'p90':>6} {'pred dur':>8}  edge")
    keys = sorted(edges[kind], key=lambda k: first[(kind, k)][1])
    n = max(len(v) for v in edges[kind].values())
    for k in keys:
        v = edges[kind][k]
        if len(v) < n // 2:
            continue
        g = sorted(x[0] for x in v)
        print(f"  {st.median(g):6.2f} {g[len(g) // 10]:6.2f} {g[9 * len(g) // 10]:6.2f} "
              f"{st.median(x[2] for x in v):8.2f}  {k[0][:34]} -> {k[1][:34]}")
