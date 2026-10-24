"""Kernel sequence of the 4-layer cycle (anchor + 3 skip layers) in the decode
verify graph, from an Nsight Systems sqlite export.

A layer starts at the all-reduce + RMSNorm that follows the previous layer's
MoE (finalize_kernel), and an anchor is a layer that runs the DSA indexer
(sm90_fp8_paged_mqa_logits). Per kernel: median duration, and the wall time it
adds to the layer: idle before it (start past every earlier kernel's end) plus
advance (its end past every earlier end); the two sum to the layer wall time.
excl: time it runs alone (no other graph kernel running). Side-branch kernels
(--side; default the skip-KV stager: anchors copy the rows their 3 skip
layers select from Grace to HBM, skip_kv_stage.py) get no idle/adds, so the
main chain's adds sum to the wall; their cost shows as slower neighbours.
Kernels are matched across layers by (name, k-th occurrence in the layer).

    cycle_nsys.py report.sqlite [--device N] [--min-kernels 500] [--json out]
"""
import argparse
import collections
import json
import re
import sqlite3
import statistics as st


def short(n):
    n = re.sub(r"^kernel_cutlass_kernel_vllmmodel_executorkernels", "cute:", n)
    return n[:58]


ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("--device", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--min-kernels", type=int, default=500)
ap.add_argument("--json")
ap.add_argument("--side", default="_mark_rows|_compact_rows|_remap_rows|_gather_rows",
                help="side-branch kernels: shown, but left out of idle/adds")
a = ap.parse_args()
side = re.compile(a.side or "^$")
c = sqlite3.connect(a.report)
names = dict(c.execute("select id, value from StringIds"))
graphs = []
for dev in a.device:
    by_corr = collections.defaultdict(list)
    for s, e, corr, n, gx, gy, gz, bx in c.execute(
            "select start, end, correlationId, shortName, gridX, gridY, gridZ, blockX "
            "from CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ? order by start", (dev,)):
        by_corr[corr].append((s, e, short(names[n]), f"{gx}x{gy}x{gz}/{bx}"))
    graphs += [ks for ks in by_corr.values() if len(ks) >= a.min_kernels]
nk = collections.Counter(len(g) for g in graphs).most_common(1)[0][0]
graphs = [g for g in graphs if len(g) == nk]


def layers(ks):
    """Split one verify graph into layers; returns [(kind, [kernel...])]."""
    cut = [0]
    for i, k in enumerate(ks):
        if k[2].startswith("allreduce_fusion") and i and any(
                "finalize_kernel" in x[2] for x in ks[max(0, i - 3):i]):
            cut.append(i)
    segs = [ks[i:j] for i, j in zip(cut, cut[1:] + [len(ks)])]
    out = []
    for seg in segs:
        anchor = any("paged_mqa_logits" in k[2] for k in seg)
        moe = any("layer_kernel" in k[2] for k in seg)
        out.append((("A" if anchor else "S") + ("" if moe else "-dense"), seg))
    return out


def exclusive(seg):
    """Time each kernel runs with no other kernel of the graph running."""
    ev = sorted([(k[0], 1, i) for i, k in enumerate(seg)] + [(k[1], -1, i) for i, k in enumerate(seg)])
    run, ex, t = set(), [0.0] * len(seg), ev[0][0]
    for tt, d, i in ev:
        if len(run) == 1:
            ex[next(iter(run))] += tt - t
        t = tt
        (run.add if d > 0 else run.discard)(i)
    return ex


def decompose(seg, t_next):
    rows, maxe = [], seg[0][0]
    ex = exclusive(seg)
    occ = collections.Counter()
    for j, (s, e, n, g) in enumerate(seg):
        occ[n] += 1
        if side.search(n):
            rows.append(((n, occ[n]), g, (e - s) / 1e3, 0.0, 0.0, (s - seg[0][0]) / 1e3, ex[j] / 1e3))
            continue
        idle = max(0, s - maxe)
        adv = max(0, e - max(maxe, s))
        rows.append(((n, occ[n]), g, (e - s) / 1e3, idle / 1e3, adv / 1e3, (s - seg[0][0]) / 1e3, ex[j] / 1e3))
        maxe = max(maxe, e)
    tail = max(0, t_next - maxe) / 1e3
    return rows, (t_next - seg[0][0]) / 1e3, tail


# per layer kind: (name, occurrence) -> lists
acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
walls = collections.defaultdict(list)
tails = collections.defaultdict(list)
cycle_walls = []
tailw = []
kinds_seq = None
for ks in graphs:
    L = layers(ks)
    kinds_seq = kinds_seq or [k for k, _ in L]
    tailw.append((ks[-1][1] - L[-1][1][0][0]) / 1e3)
    starts = [seg[0][0] for _, seg in L] + [ks[-1][1]]
    # position in the anchor cycle: A, S1, S2, S3
    # segment 0 = dense layers 0-2 + layer 3 (S1 of the anchor at layer 2)
    pos, kinds = 1, ["prologue"]
    for k, _ in L[1:]:
        pos = 0 if k.startswith("A") else pos + 1
        kinds.append(k if k.endswith("dense") else ("A" if pos == 0 else f"S{pos}"))
    for li, ((_, seg), kind) in enumerate(zip(L, kinds)):
        if li == len(L) - 1:
            continue  # last segment holds the final norm / step tail
        rows, wall, tail = decompose(seg, starts[li + 1])
        walls[kind].append(wall)
        tails[kind].append(tail)
        for key, g, d, idle, adv, off, ex in rows:
            r = acc[kind][key]
            r["dur"].append(d); r["idle"].append(idle); r["adv"].append(adv)
            r["off"].append(off); r["grid"].append(g); r["ex"].append(ex)
    # full cycles: A S1 S2 S3 with MoE
    for i in range(len(kinds) - 4):
        if kinds[i:i + 4] == ["A", "S1", "S2", "S3"]:
            cycle_walls.append((starts[i + 4] - starts[i]) / 1e3)

span = st.median((g[-1][1] - g[0][0]) / 1e3 for g in graphs)
print(f"devices {a.device}: {len(graphs)} verify graphs of {nk} kernels, median span {span:.0f} us")
print("segments:", " ".join(k[0] if not k.endswith("dense") else k[0].lower() for k in kinds_seq),
      "(first = dense layers 0-2 + layer 3, last = final norm)")
print(f"prologue wall {st.median(walls['prologue']):.1f} us, final segment {st.median(tailw):.1f} us")
print(f"median 4-layer cycle (A S1 S2 S3) wall: {st.median(cycle_walls):.1f} us "
      f"(p10 {sorted(cycle_walls)[len(cycle_walls) // 10]:.1f}, "
      f"p90 {sorted(cycle_walls)[9 * len(cycle_walls) // 10]:.1f}, n {len(cycle_walls)})")
res = {}
for kind in ["A", "S1", "S2", "S3"]:
    if kind not in walls:
        continue
    print(f"\n== {kind}: median wall {st.median(walls[kind]):.1f} us over {len(walls[kind])} layers"
          f" (tail idle to next layer {st.median(tails[kind]):.1f})")
    print(f"  {'#':>3} {'start':>7} {'dur':>7} {'idle':>5} {'adds':>6} {'excl':>6}  {'grid/blk':14s} kernel")
    tot = 0.0
    res[kind] = []
    keys = [k for k in acc[kind] if len(acc[kind][k]["dur"]) >= len(walls[kind]) // 2]
    keys.sort(key=lambda k: st.median(acc[kind][k]["off"]))
    for i, key in enumerate(keys):
        r = acc[kind][key]
        d, idle, adv, off, ex = (st.median(r[x]) for x in ("dur", "idle", "adv", "off", "ex"))
        g = collections.Counter(r["grid"]).most_common(1)[0][0]
        tot += idle + adv
        n = key[0] + (f" #{key[1]}" if key[1] > 1 else "")
        print(f"  {i:3d} {off:7.1f} {d:7.2f} {idle:5.1f} {adv:6.2f} {ex:6.2f}  {g:14s} {n}")
        res[kind].append(dict(kernel=key[0], occ=key[1], grid=g, start=off, dur=d, idle=idle, adds=adv, excl=ex,
                              n=len(r["dur"])))
    print(f"  sum of medians (idle + adds) {tot:.1f} us")
if a.json:
    json.dump(dict(span=span, cycle=st.median(cycle_walls), walls={k: st.median(v) for k, v in walls.items()},
                   layers=res), open(a.json, "w"), indent=1)
