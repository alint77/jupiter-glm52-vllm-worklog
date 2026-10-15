"""Per-engine-step timeline of one GPU from an nsys report (graph kernels traced
per node): steps start at Model Runner V2's input-prep kernel
(_compute_slot_mappings_kernel). Per step: kind (decode = verify full graph,
prefill = tiered prefill MoE kernels), wall, GPU busy/idle (union over
streams), idle gaps and what the worker's CUDA API thread was doing in them,
kernel time by group, memcpy bytes by kind.

    step_timeline.py report.sqlite [--device 0] [--steps] [--gaps N]
"""
import argparse
import collections
import sqlite3

GROUPS = [
    ("prep(MRv2 input)", ("_compute_slot", "_gather_block_tables", "_combine_sampled",
                          "_apply_write", "_dcp_local_seq", "_build_prefill_chunk",
                          "DeviceScan", "_prepare", "_post_update", "_expand")),
    ("moe_prefill", ("gemm_kernel", "gemm_xres", "tiered_prefill", "route_kernel",
                     "act_kernel", "combine_kernel", "gather_kernel", "finalize_kernel")),
    ("moe_decode", ("layer_kernel", "decode_gemm")),
    ("drafter_moe", ("Marlin",)),
    ("nccl_allgather", ("AllGather",)),
    ("nccl_allreduce", ("AllReduce",)),
    ("ar_fused", ("allreduce_fusion", "cross_device_reduce")),
    ("attn_prefill", ("sparse_attn_fwd",)),
    ("attn_decode", ("flash_fwd_splitkv_mla", "flash_fwd_mla_combine", "flash_mla")),
    ("dcp_combine", ("lse_combine", "gather_cat", "lse_reduce", "dcp_")),
    ("indexer", ("mqa_logits", "topKPerRow", "indexer", "TopK", "_pack_dcp_topk",
                 "_convert_req_index")),
    ("kv_gather/cache", ("cp_gather", "concat_and_cache", "reshape_and_cache")),
    ("dense_gemm", ("nvjet", "cutlass", "sm90_xmma", "gemm")),
    ("norm/elementwise", ("rms_norm", "triton_", "elementwise", "Fill", "Cat", "copy",
                          "reduce_kernel", "index")),
    ("sampler", ("sample", "topk", "softmax", "argmax", "rejection", "gumbel")),
    ("barrier", ("barrier_kernel",)),
]
KINDS = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}


def group(n):
    for g, keys in GROUPS:
        if any(k in n for k in keys):
            return g
    return "other"


def union(iv):
    tot = 0
    end = None
    start = None
    for s, e in sorted(iv):
        if end is None or s > end:
            if end is not None:
                tot += end - start
            start, end = s, e
        else:
            end = max(end, e)
    if end is not None:
        tot += end - start
    return tot


def gaps(iv, lo, hi, minlen):
    out = []
    cur = lo
    for s, e in sorted(iv):
        if s > cur and s - cur >= minlen:
            out.append((cur, s))
        cur = max(cur, e)
    if hi - cur >= minlen:
        out.append((cur, hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--steps", action="store_true")
    ap.add_argument("--gaps", type=int, default=0, help="print N largest gaps")
    ap.add_argument("--min-gap", type=float, default=50.0, help="us")
    a = ap.parse_args()
    c = sqlite3.connect(a.report)
    names = dict(c.execute("select id, value from StringIds"))
    pid = c.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? "
                    "limit 1", (a.device,)).fetchone()[0]
    ks = [(s, e, names[n], corr) for s, e, n, corr in c.execute(
        "select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL "
        "where deviceId=? order by start", (a.device,))]
    mc = c.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY "
                   "where deviceId=? order by start", (a.device,)).fetchall()
    ms = c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET where deviceId=?",
                   (a.device,)).fetchall()
    rt = [(s, e, names[n], t & 0xFFFFFF, corr) for s, e, t, corr, n in c.execute(
        "select start, end, globalTid, correlationId, nameId from "
        "CUPTI_ACTIVITY_KIND_RUNTIME where (globalTid & ~16777215)=? order by start",
        (pid,))]
    main_tid = collections.Counter(r[3] for r in rt).most_common(1)[0][0]
    rt = [r for r in rt if r[3] == main_tid]
    per_corr = collections.Counter(k[3] for k in ks)
    marks = [k[0] for k in ks if k[2].startswith("_compute_slot_mappings")]
    T0 = ks[0][0]
    bounds = list(zip(marks, marks[1:]))
    steps = []
    import bisect
    kstarts = [k[0] for k in ks]
    mstarts = [m[0] for m in mc]
    rstarts = [r[0] for r in rt]
    for lo, hi in bounds:
        # include the prep kernels just before the marker (same step)
        i0 = bisect.bisect_left(kstarts, lo - 3000_000)
        i1 = bisect.bisect_left(kstarts, hi)
        sk = [k for k in ks[i0:i1] if k[0] >= lo]
        j0, j1 = bisect.bisect_left(mstarts, lo), bisect.bisect_left(mstarts, hi)
        sm = mc[j0:j1]
        r0, r1 = bisect.bisect_left(rstarts, lo), bisect.bisect_left(rstarts, hi)
        sr = rt[r0:r1]
        big = max((per_corr[k[3]] for k in sk), default=0)
        kind = ("prefill" if any(k[2] in ("gemm_kernel", "gemm_xres_kernel") for k in sk)
                else "decode" if big >= 1500 else "other")
        iv = [(k[0], k[1]) for k in sk] + [(m[0], m[1]) for m in sm]
        busy = union([(max(s, lo), min(e, hi)) for s, e in iv if e > lo and s < hi])
        g = collections.Counter()
        for k in sk:
            g[group(k[2])] += k[1] - k[0]
        mb = collections.Counter()
        mcount = collections.Counter()
        for m in sm:
            mb[KINDS.get(m[3], str(m[3]))] += m[2]
            mcount[KINDS.get(m[3], str(m[3]))] += 1
        gp = gaps(iv, lo, hi, a.min_gap * 1000)
        api = collections.Counter()
        for r in sr:
            api[r[2].split("_v")[0]] += r[1] - r[0]
        steps.append(dict(lo=lo, hi=hi, kind=kind, wall=hi - lo, busy=busy, groups=g,
                          mbytes=mb, mcount=mcount, gaps=gp, api=api, nk=len(sk),
                          rt=sr))
    print(f"device {a.device}, {len(steps)} steps, window {(ks[-1][1] - T0) / 1e6:.0f} ms")
    if a.steps:
        print("  t(ms)   kind     wall   busy  idle  gaps>=%dus  H2D MB  D2H n  kernels"
              % a.min_gap)
        for s in steps:
            print(f"  {(s['lo'] - T0) / 1e6:7.1f} {s['kind']:8} {s['wall'] / 1e6:6.2f} "
                  f"{s['busy'] / 1e6:6.2f} {(s['wall'] - s['busy']) / 1e6:5.2f} "
                  f"{len(s['gaps']):4d}  {s['mbytes']['H2D'] / 2**20:8.1f} "
                  f"{s['mcount']['D2H']:4d}  {s['nk']}")
    for kind in ("decode", "prefill", "other"):
        ss = [s for s in steps if s["kind"] == kind]
        if not ss:
            continue
        n = len(ss)
        wall = sum(s["wall"] for s in ss) / n
        busy = sum(s["busy"] for s in ss) / n
        print(f"\n== {kind}: {n} steps, mean wall {wall / 1e6:.2f} ms, GPU busy "
              f"{busy / 1e6:.2f} ms, idle {(wall - busy) / 1e6:.2f} ms "
              f"({100 * (wall - busy) / wall:.0f}%)")
        g = collections.Counter()
        for s in ss:
            g.update(s["groups"])
        print("   kernel time by group (ms/step, summed over streams):")
        for k, v in g.most_common():
            print(f"     {v / n / 1e6:8.3f}  {k}")
        mb = collections.Counter()
        mn = collections.Counter()
        for s in ss:
            mb.update(s["mbytes"])
            mn.update(s["mcount"])
        print("   memcpy per step: " + ", ".join(
            f"{k} {mn[k] / n:.1f}x {mb[k] / n / 2**20:.2f} MiB" for k in mb))
        api = collections.Counter()
        for s in ss:
            api.update(s["api"])
        print("   worker main-thread CUDA API (ms/step):")
        for k, v in api.most_common(10):
            print(f"     {v / n / 1e6:8.3f}  {k}")
        gl = sorted(((e - s, s, e, st) for st in ss for s, e in st["gaps"]), reverse=True)
        tot_gap = sum(x[0] for x in gl) / n
        print(f"   idle gaps >= {a.min_gap:.0f} us: {sum(len(s['gaps']) for s in ss) / n:.1f}/step, "
              f"{tot_gap / 1e6:.2f} ms/step")
        for d, s, e, st in gl[:a.gaps]:
            calls = collections.Counter()
            for r in st["rt"]:
                ov = min(r[1], e) - max(r[0], s)
                if ov > 0:
                    calls[r[2].split("_v")[0]] += ov
            print(f"     gap {d / 1e3:8.1f} us at +{(s - st['lo']) / 1e6:6.2f} ms of step "
                  f"(t={(s - T0) / 1e6:.1f}): CPU in " + ", ".join(
                      f"{k} {v / 1e3:.0f}us" for k, v in calls.most_common(4)))


if __name__ == "__main__":
    main()
