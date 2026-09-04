import collections, gzip, json, statistics
from pathlib import Path
GPU={"kernel","gpu_memcpy","gpu_memset"}; LAU={"cuda_runtime","cuda_driver"}
p=sorted(Path("/e/project1/profound/alint77/traces/mtp3-profile-1665068/prefill").glob("*_rank0.*trace.json.gz"))[0]
with gzip.open(p,"rt") as fh: ev=[e for e in json.load(fh)["traceEvents"] if e.get("ph")=="X"]
o=min(e["ts"] for e in ev if e.get("cat") in GPU)
for e in ev: e["t"]=(e["ts"]-o)/1000
cpu=collections.defaultdict(list)
for e in ev:
    if e.get("cat")=="cpu_op": cpu[e["args"].get("External id")].append(e)
bc=collections.defaultdict(list)
for e in ev:
    if e.get("cat") in GPU and e.get("args",{}).get("correlation") is not None: bc[e["args"]["correlation"]].append(e)
lau=sorted((e for e in ev if e.get("cat") in LAU),key=lambda e:e["t"])
ann=sorted((e for e in ev if e.get("cat")=="user_annotation" and e["name"].startswith("execute_")),key=lambda e:e["t"])
w=[(a["t"],b["t"]) for a,b in zip(ann,ann[1:])]
s,en=w[len(w)//2]
corr=[e["args"]["correlation"] for e in lau if s<=e["t"]<en and "correlation" in e.get("args",{})]
ops=sorted((k for c in corr for k in bc[c]),key=lambda e:e["t"])
def inner(x):
    l=[q for q in cpu.get(x,[]) if q["cat"]=="cpu_op"]
    return min(l,key=lambda q:q["dur"]) if l else None
rows=[]
for k in ops:
    if "marlin_moe_wna16" not in k["name"]: continue
    op=inner(k["args"].get("External id"))
    if not op: continue
    d=op["args"]["Input Dims"]; a,bq,sc=d[0],d[2],d[4]
    gemm="w13" if a[1]==6144 else "w2"
    by=bq[0]*bq[1]*bq[2]*4 + sc[0]*sc[1]*sc[2]*2
    rows.append({"gemm":gemm,"E":bq[0],"us":k["dur"],"bytes":by,"bw":by/(k["dur"]*1e-6)/1e9})
print(f"{len(rows)} marlin calls in one chunk")
for gemm in ("w13","w2"):
    r=[x for x in rows if x["gemm"]==gemm]
    bws=sorted(x["bw"] for x in r)
    print(f"\n{gemm}: n={len(r)}  effective weight BW GB/s: "
          f"min {bws[0]:.0f} p25 {bws[len(bws)//4]:.0f} p50 {bws[len(bws)//2]:.0f} "
          f"p75 {bws[3*len(bws)//4]:.0f} max {bws[-1]:.0f}")
    hist=collections.Counter(int(x["bw"]//100)*100 for x in r)
    print("   BW histogram (GB/s bucket: count):", dict(sorted(hist.items())))
# pair per layer
print("\nper-layer pairs (w13), first 14 layers: (E, us, GB/s)")
w13=[x for x in rows if x["gemm"]=="w13"]
for i in range(0,min(28,len(w13)),2):
    a,b=w13[i],w13[i+1]
    print(f"  L{i//2:2d}: ({a['E']:2d}E {a['us']:7.0f}us {a['bw']:5.0f}) "
          f"({b['E']:2d}E {b['us']:7.0f}us {b['bw']:5.0f})   sum E={a['E']+b['E']}")
