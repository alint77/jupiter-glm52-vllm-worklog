"""Cold-prefetch H2D copy bandwidth vs the kernels running during each copy.

    copy_overlap.py report.sqlite [--device N] [--min-mib 1]

Per device: every H2D memcpy >= min-mib, its bandwidth, and the fraction of
its duration overlapped by each kernel (any stream on that device). Prints the
kernels that overlap copies most, then copy bandwidth bucketed by how much of
the copy a host-KV kernel (KV write into Grace, KV gather from Grace) covered.
"""
import argparse, bisect, collections, re, sqlite3, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("--device", type=int, default=None)
ap.add_argument("--min-mib", type=float, default=1)
a = ap.parse_args()
c = sqlite3.connect(a.report)
names = dict(c.execute("select id, value from StringIds"))
devs = [a.device] if a.device is not None else [
    d for (d,) in c.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_MEMCPY "
                            "where copyKind=1 and bytes>=? order by 1", (int(a.min_mib * 2**20),))]
HOSTKV = re.compile(r"concat_and_cache|cp_gather|indexer_k_quant_and_cache|gather_cache")
ATTN = re.compile(r"flash_fwd|sparse|fmha|mla_combine|TopK")
MOE = re.compile(r"tiered_prefill|tiered_decode|marlin|moe|^gemm_kernel|^gemm_xres_kernel|^route_kernel|^gather_kernel|^combine_kernel|^act_kernel|^prep", re.I)
COMM = re.compile(r"nccl|allreduce|one_shot|cross_device", re.I)

def cls(n):
    for k, r in (("hostkv", HOSTKV), ("attn", ATTN), ("moe", MOE), ("comm", COMM)):
        if r.search(n): return k
    return "other"

for dev in devs:
    ks = [(s, e, names[n]) for s, e, n in c.execute(
        "select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start", (dev,))]
    starts = [k[0] for k in ks]
    maxdur = max(e - s for s, e, _ in ks)
    cps = c.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? and copyKind=1 "
                    "and bytes>=? order by start", (dev, int(a.min_mib * 2**20))).fetchall()
    by_name = collections.Counter(); rows = []
    for cs, ce, b in cps:
        i = bisect.bisect_left(starts, cs - maxdur)
        frac = collections.Counter()
        while i < len(ks) and ks[i][0] < ce:
            s, e, n = ks[i]
            ov = min(e, ce) - max(s, cs)
            if ov > 0:
                frac[cls(n)] += ov; by_name[n] += ov
            i += 1
        d = ce - cs
        rows.append((b / d, b, {k: v / d for k, v in frac.items()}))
    tot = sum(ce - cs for cs, ce, _ in cps)
    print(f"== device {dev}: {len(cps)} H2D copies >= {a.min_mib} MiB, {sum(b for *_, b in cps)/2**30:.1f} GiB, "
          f"busy {tot/1e6:.1f} ms, median {st.median(r[0] for r in rows):.0f} GB/s, "
          f"bytes-weighted {sum(b for *_, b in cps)/tot:.0f} GB/s")
    print("   kernels overlapping copies most (ms of overlap):")
    for n, v in by_name.most_common(10): print(f"   {v/1e6:8.2f}  {cls(n):6s} {n[:80]}")
    def show(label, sel):
        if not sel: print(f"   {label:34s} n=0"); return
        bw = sorted(r[0] for r in sel); b = sum(r[1] for r in sel)
        print(f"   {label:34s} n={len(sel):4d} {b/2**30:6.1f} GiB  median {st.median(bw):4.0f}  p10 {bw[len(bw)//10]:4.0f}  p90 {bw[(9*len(bw))//10]:4.0f} GB/s")
    show("hostkv overlap = 0", [r for r in rows if r[2].get("hostkv", 0) == 0])
    show("hostkv overlap 0-20%", [r for r in rows if 0 < r[2].get("hostkv", 0) < 0.2])
    show("hostkv overlap >= 20%", [r for r in rows if r[2].get("hostkv", 0) >= 0.2])
    show("attn overlap >= 50%", [r for r in rows if r[2].get("attn", 0) >= 0.5])
    show("moe overlap >= 50%", [r for r in rows if r[2].get("moe", 0) >= 0.5])
    show("comm overlap >= 50%", [r for r in rows if r[2].get("comm", 0) >= 0.5])
    show("nothing running (< 5% busy)", [r for r in rows if sum(r[2].values()) < 0.05])
