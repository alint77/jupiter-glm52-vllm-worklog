"""Skip-layer KV staging (_mark/_remap/_compact/_gather_rows: UVA reads of the
Grace-resident KV inside the verify graph) per decode step: kernel time,
the union of its intervals, and the part with no other kernel running on the
device (exposed); which kernels overlap it, and whether it shares a stream
with them. Steps = verify graphs (>= 500 kernels per correlation id).

    stage_exposed.py report.sqlite [--device 0]
"""
import argparse
import collections
import sqlite3

STAGE = ("_gather_rows", "_remap_rows", "_compact_rows", "_mark_rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    c = sqlite3.connect(a.db)
    names = dict(c.execute("select id, value from StringIds"))
    ks = [(s, e, names[n], st, corr) for s, e, n, st, corr in c.execute(
        "select start, end, shortName, streamId, correlationId from CUPTI_ACTIVITY_KIND_KERNEL "
        "where deviceId=? order by start", (a.device,))]
    per = collections.Counter(k[4] for k in ks)
    steps = collections.defaultdict(list)
    for k in ks:
        if per[k[4]] >= 500:
            steps[k[4]].append(k)
    tot = collections.Counter()
    over = collections.Counter()
    same = diff = 0
    n = 0
    for corr, sk in steps.items():
        stg = [k for k in sk if k[2].startswith(STAGE)]
        if not stg:
            continue
        n += 1
        lo, hi = sk[0][0], max(k[1] for k in sk)
        oth = [k for k in ks if k[1] > lo and k[0] < hi and not k[2].startswith(STAGE)]
        tot["step wall"] += hi - lo
        tot["stage kernel time"] += sum(k[1] - k[0] for k in stg)
        tot["gather kernel time"] += sum(k[1] - k[0] for k in stg if k[2].startswith("_gather"))
        # sweep: stage active vs other active
        evs = sorted([(k[0], 1, 0) for k in stg] + [(k[1], -1, 0) for k in stg]
                     + [(k[0], 0, 1) for k in oth] + [(k[1], 0, -1) for k in oth])
        cs = co = 0
        last = None
        for t, ds, do in evs:
            if last is not None and cs > 0:
                tot["stage union"] += t - last
                if co == 0:
                    tot["stage exposed (nothing else running)"] += t - last
            cs += ds
            co += do
            last = t
        for g in stg:
            for k in oth:
                ov = min(g[1], k[1]) - max(g[0], k[0])
                if ov > 0:
                    over[k[2][:60]] += ov
                    if k[3] == g[3]:
                        same += ov
                    else:
                        diff += ov
    print(f"device {a.device}: {n} verify steps with staging")
    for k in ("step wall", "stage kernel time", "gather kernel time", "stage union",
              "stage exposed (nothing else running)"):
        print(f"  {k:<40} {tot[k] / n / 1e3:8.1f} us/step")
    print(f"  overlap with other kernels: {(diff) / n / 1e3:.1f} us/step on other streams")
    print("  kernels overlapping the staging (us/step):")
    for k, v in over.most_common(8):
        print(f"    {v / n / 1e3:7.1f}  {k}")


if __name__ == "__main__":
    main()
