import sqlite3, collections, statistics as st, sys
R="/e/fscratch/profound/naeimitabiei1/sliced-prof/"
def per_step(db):
    c=sqlite3.connect(db); names=dict(c.execute("select id,value from StringIds"))
    per=collections.defaultdict(list)
    for s,e,n,cid in c.execute("select start,end,shortName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0"):
        per[cid].append((s,e,names[n]))
    out=[]
    for v in per.values():
        if len(v)<500: continue
        d=collections.Counter()
        for s,e,n in v: d[n]+=(e-s)/1e3
        out.append(d)
    return out
for hold,kind,win in [(h,k,w) for h,k in [("2413889","dec4"),("2413892","dec4"),("2413893","tlnsys"),("2413890","tlnsys")] for w in ((1,2) if k=="dec4" else (2,))]:
    A=per_step(f"{R}aka-{kind}-{hold}/nsys.{win}.sqlite"); B=per_step(f"{R}akb-{kind}-{hold}/nsys.{win}.sqlite")
    keys=set().union(*A,*B)
    diff={k:st.median(b[k] for b in B)-st.median(a[k] for a in A) for k in keys}
    print(f"{kind} {hold} win{win}: steps {len(A)}/{len(B)}")
    for k,v in sorted(diff.items(),key=lambda t:-abs(t[1]))[:6]:
        print(f"   {v:+8.1f} us  a {st.median(a[k] for a in A):8.1f}  {k[:70]}")
