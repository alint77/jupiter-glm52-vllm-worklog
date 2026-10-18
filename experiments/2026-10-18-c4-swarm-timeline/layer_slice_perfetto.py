"""One anchor layer + its 3 index-sharing (skip) layers of one verify step, all
GPUs, as a Chrome trace for Perfetto: kernels (and memcpys) only, per device
and stream. Layers are cut at the end of each MoE (the finalize_kernel after a
layer_kernel); an anchor is a layer whose span runs the indexer
(paged_mqa_logits). The same step ordinal and layer index on every device;
side-stream kernels (indexer, skip-KV staging) overlapping the span are
included.

    layer_slice_perfetto.py report.sqlite out.json.gz [--step-frac 0.5] [--near-layer 30]
"""
import argparse
import collections
import gzip
import json
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("out")
    ap.add_argument("--step-frac", type=float, default=0.5)
    ap.add_argument("--near-layer", type=int, default=30)
    a = ap.parse_args()
    c = sqlite3.connect(a.db)
    names = dict(c.execute("select id, value from StringIds"))
    devs = [d for (d,) in c.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL")]
    spans, kern = {}, {}
    pick = None
    for d in sorted(devs):
        ks = [(s, e, names[n], st, corr, gx, gy, gz, bx, by, bz) for s, e, n, st, corr, gx, gy, gz, bx, by, bz
              in c.execute("select start, end, shortName, streamId, correlationId, gridX, gridY, "
                           "gridZ, blockX, blockY, blockZ from CUPTI_ACTIVITY_KIND_KERNEL "
                           "where deviceId=? order by start", (d,))]
        kern[d] = ks
        per = collections.Counter(k[4] for k in ks)
        order = sorted({k[4] for k in ks if per[k[4]] >= 500},
                       key=lambda x: min(k[0] for k in ks if k[4] == x))
        corr = order[int(a.step_frac * (len(order) - 1))]
        sk = [k for k in ks if k[4] == corr]
        fin = [k for k in sk if k[2] == "finalize_kernel"]
        idx = [k[0] for k in sk if "paged_mqa_logits" in k[2] and "metadata" not in k[2]]
        cuts = [f[1] for f in fin]
        anchor = [any(lo < t < hi for t in idx) for lo, hi in zip(cuts, cuts[1:])]
        if pick is None:
            cand = [i for i in range(len(anchor) - 3)
                    if anchor[i] and not any(anchor[i + 1:i + 4])]
            pick = min(cand, key=lambda i: abs(i - a.near_layer))
        spans[d] = (cuts[pick], cuts[pick + 4])
        print(f"GPU {d}: step corr {corr}, MoE-layer cut {pick}..{pick + 4}, "
              f"{(spans[d][1] - spans[d][0]) / 1e3:.1f} us, anchors {anchor[pick:pick + 4]}")
    t0 = min(s for s, _ in spans.values())
    ev = []
    for d in sorted(devs):
        lo, hi = spans[d]
        for s, e, n, st, corr, gx, gy, gz, bx, by, bz in kern[d]:
            if e > lo and s < hi:
                ev.append({"ph": "X", "name": n, "pid": d, "tid": st, "ts": (s - t0) / 1e3,
                           "dur": (e - s) / 1e3, "cat": "kernel",
                           "args": {"grid": f"{gx}x{gy}x{gz}", "block": f"{bx}x{by}x{bz}"}})
        for s, e, st, b, k in c.execute(
                "select start, end, streamId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY "
                "where deviceId=? and end>? and start<?", (d, lo, hi)):
            ev.append({"ph": "X", "name": f"memcpy {k} {b} B", "pid": d, "tid": st,
                       "ts": (s - t0) / 1e3, "dur": (e - s) / 1e3, "cat": "memcpy"})
        ev.append({"ph": "M", "name": "process_name", "pid": d, "args": {"name": f"GPU {d}"}})
        ev.append({"ph": "M", "name": "process_sort_index", "pid": d, "args": {"sort_index": d}})
    with gzip.open(a.out, "wt") as f:
        json.dump({"traceEvents": ev, "displayTimeUnit": "ns"}, f)
    print(f"{len(ev)} events -> {a.out}")


if __name__ == "__main__":
    main()
