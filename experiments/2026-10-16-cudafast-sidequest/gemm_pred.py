"""Per verify graph: every dense GEMM call (decode_gemm, nvjet, splitK), its
same-stream predecessor kernel, gap and own duration."""
import sqlite3, sys, collections, re, statistics as st
db, dev = sys.argv[1], int(sys.argv[2])
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
def short(n):
    n = re.sub(r"\(.*", "", n); n = re.sub(r"<.*", "", n)
    return n.split("::")[-1].replace("void ", "")[:44]
g = collections.defaultdict(list)
for s, e, corr, sid, n, gx in c.execute("select start,end,correlationId,streamId,demangledName,gridX from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=?", (dev,)):
    g[corr].append((s, e, sid, short(names[n]), gx))
graphs = [ks for ks in g.values() if len(ks) >= 500]
isg = lambda n: n.startswith("decode_gemm") or n.startswith("nvjet")
stat = collections.defaultdict(lambda: [0, [], []])
calls = collections.Counter()
for ks in graphs:
    by = collections.defaultdict(list)
    for k in ks: by[k[2]].append(k)
    for sid, m in by.items():
        m.sort()
        for a, b in zip(m, m[1:]):
            if isg(b[3]):
                key = (b[3][:20], a[3])
                stat[key][0] += 1; stat[key][1].append(b[0] - a[1]); stat[key][2].append(b[1] - b[0])
    for k in ks:
        if isg(k[3]): calls[k[3][:20]] += 1
G = len(graphs)
print("GEMM calls per graph:", {k: round(v / G, 1) for k, v in calls.items()})
print(f"{'n/graph':>7} {'gap_ns':>7} {'dur_us':>7} {'sum_us':>7}  gemm <- predecessor")
for k, (n, gaps, durs) in sorted(stat.items(), key=lambda kv: -sum(kv[1][2])):
    print(f"{n/G:7.1f} {st.median(gaps):7.0f} {st.median(durs)/1e3:7.2f} {sum(durs)/G/1e3:7.0f}  {k[0]} <- {k[1]}")
