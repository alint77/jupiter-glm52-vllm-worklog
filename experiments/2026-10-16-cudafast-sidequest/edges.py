"""Verify-graph main-stream edges: for consecutive kernels on the busiest stream
of each >=500-kernel graph, gap = next.start - prev.end (negative = overlap,
i.e. PDL). Aggregate per (prev, next) short-name pair: count/graph, median gap,
total gap per graph."""
import sqlite3, sys, collections, statistics as st, re
db, dev = sys.argv[1], int(sys.argv[2])
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
def short(n):
    n = re.sub(r"\(.*", "", n); n = re.sub(r"<.*", "", n)
    return n.split("::")[-1].replace("void ", "")[:48]
g = collections.defaultdict(list)
for s, e, corr, sid, n in c.execute("select start,end,correlationId,streamId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=?", (dev,)):
    g[corr].append((s, e, sid, names[n]))
graphs = [ks for ks in g.values() if len(ks) >= 500]
pair = collections.defaultdict(list)
tot_gap, tot_pos = [], []
streams = collections.Counter()
for ks in graphs:
    sc = collections.Counter(k[2] for k in ks); main = sc.most_common(1)[0][0]
    m = sorted(k for k in ks if k[2] == main)
    tg = 0; tp = 0
    for a, b in zip(m, m[1:]):
        gap = b[0] - a[1]
        pair[(short(a[3]), short(b[3]))].append(gap)
        tg += gap; tp += max(0, gap)
    tot_gap.append(tg); tot_pos.append(tp)
    streams[len(sc)] += 1
G = len(graphs)
print(f"{G} graphs; main-stream kernels/graph {len(m)}; streams/graph {dict(streams)}")
print(f"sum of positive gaps on main stream per graph: median {st.median(tot_pos)/1e3:.0f} us; net (incl overlap) {st.median(tot_gap)/1e3:.0f} us")
allg = sorted(x for v in pair.values() for x in v)
print("gap distribution (ns): p10 %d p50 %d p90 %d" % (allg[len(allg)//10], allg[len(allg)//2], allg[len(allg)*9//10]))
rows = sorted(pair.items(), key=lambda kv: -sum(max(0,x) for x in kv[1]))
print(f"{'per graph':>9} {'med_ns':>7} {'pos_us/graph':>12}  prev -> next")
for (a, b), v in rows[:40]:
    print(f"{len(v)/G:9.1f} {st.median(v):7.0f} {sum(max(0,x) for x in v)/G/1e3:12.1f}  {a} -> {b}")
