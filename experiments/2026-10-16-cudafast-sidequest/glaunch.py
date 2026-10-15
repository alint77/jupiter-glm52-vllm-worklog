"""Per cudaGraphLaunch: host API duration, node count, and whether the GPU was
idle when the launch call started (prev kernel end on that device)."""
import sqlite3, sys, collections, statistics as st
db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
gl = [k for k, v in names.items() if v.startswith("cudaGraphLaunch")]
ks = collections.defaultdict(list)
allk = []
for s, e, corr in c.execute("select start,end,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=?", (dev,)):
    ks[corr].append((s, e)); allk.append((s, e))
allk.sort()
import bisect
ends = []
mx = 0
for s, e in allk:
    mx = max(mx, e); ends.append(mx)
starts = [s for s, _ in allk]
rows = []
q = f"select start,end,correlationId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where nameId in ({','.join(map(str,gl))})"
for s, e, corr, tid in c.execute(q):
    k = ks.get(corr)
    if not k: continue
    f = min(x[0] for x in k)
    i = bisect.bisect_left(starts, f) - 1
    prev_end = ends[i] if i >= 0 else 0
    rows.append((len(k), (e - s) / 1e3, (f - s) / 1e3, (f - prev_end) / 1e3, tid))
by = collections.defaultdict(list)
for r in rows:
    b = r[0] // 100 * 100
    by[b].append(r)
print("nodes~  n  host_us(med)  launch->1stkernel_us(med)  gpu_idle_before_us(med,p90)")
for b in sorted(by):
    r = by[b]
    idle = sorted(max(0, x[3]) for x in r)
    print(f"{b:5d} {len(r):4d}  {st.median(x[1] for x in r):8.1f}  {st.median(x[2] for x in r):10.1f}  {st.median(idle):8.1f} {idle[len(idle)*9//10]:8.1f}")
