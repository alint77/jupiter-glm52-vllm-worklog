"""Device-level idle gaps (no kernel or memcpy on any stream) in the trace,
grouped by (kernel before, kernel after); per verify-graph period."""
import sqlite3, sys, collections, re, statistics as st
db, dev = sys.argv[1], int(sys.argv[2])
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
def short(n):
    n = re.sub(r"\(.*", "", n); n = re.sub(r"<.*", "", n)
    return n.split("::")[-1].replace("void ", "")[:44]
ev = [(s, e, short(names[n]), corr) for s, e, n, corr in c.execute("select start,end,demangledName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=?", (dev,))]
ev += [(s, e, "memcpy", corr) for s, e, corr in c.execute("select start,end,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=?", (dev,))]
ev.sort()
cnt = collections.Counter(x[3] for x in ev)
gstart = sorted(min(x[0] for x in ev if x[3] == k) for k, v in cnt.items() if v >= 500)
lo, hi = gstart[0], gstart[-1]
periods = len(gstart) - 1
gaps = collections.defaultdict(list)
mx, last = 0, None
for s, e, n, _ in ev:
    if s > mx and last and lo <= s <= hi:
        gaps[(last, n)].append(s - mx)
    if e > mx: mx, last = e, n
tot = sum(sum(v) for v in gaps.values())
print(f"{periods} periods, mean period {(hi-lo)/periods/1e3:.0f} us, device idle {tot/periods/1e3:.0f} us/period")
for (a, b), v in sorted(gaps.items(), key=lambda kv: -sum(kv[1]))[:15]:
    print(f"{sum(v)/periods/1e3:7.1f} us/period  n/period {len(v)/periods:5.1f}  med {st.median(v)/1e3:6.1f} us  {a} -> {b}")
