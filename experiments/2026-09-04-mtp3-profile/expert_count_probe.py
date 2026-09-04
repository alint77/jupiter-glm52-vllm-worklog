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
s,en=w[2]
corr=[e["args"]["correlation"] for e in lau if s<=e["t"]<en and "correlation" in e.get("args",{})]
ops=sorted((k for c in corr for k in bc[c]),key=lambda e:e["t"])
seq={"w13":[],"w2":[]}
for k in ops:
    if "marlin_moe_wna16" not in k["name"]: continue
    l=[q for q in cpu.get(k["args"].get("External id"),[]) if q["cat"]=="cpu_op"]
    if not l: continue
    d=min(l,key=lambda q:q["dur"])["args"]["Input Dims"]
    a,bq,sc=d[0],d[2],d[4]
    by=bq[0]*bq[1]*bq[2]*4+sc[0]*sc[1]*sc[2]*2
    seq["w13" if a[1]==6144 else "w2"].append((bq[0],k["dur"],by))
for gemm in ("w13","w2"):
    r=seq[gemm]; hot=[];cold=[]
    for i in range(0,len(r)-1,2):
        a,b=r[i],r[i+1]
        if a[0]+b[0]!=64: continue
        hot.append(a); cold.append(b)
    print(f"\n{gemm}: {len(hot)} layer pairs")
    for nm,t in (("hot(1st)",hot),("cold(2nd)",cold)):
        E=[x[0] for x in t]; us=sum(x[1] for x in t)/1000; by=sum(x[2] for x in t)
        print(f"  {nm:9s} mean E {statistics.fmean(E):5.1f} (range {min(E)}-{max(E)}) "
              f"total E {sum(E):5d}  time {us:7.1f} ms  weight bytes {by/1e9:6.2f} GB  "
              f"-> {by/(us*1e-3)/1e9:6.0f} GB/s   {us/sum(E)*1000:5.0f} us/expert")
    # first vs second half of layers
    h=len(hot)//2
    print(f"  hot E: layers 0-{h-1} mean {statistics.fmean([x[0] for x in hot[:h]]):.1f}, "
          f"layers {h}-{len(hot)-1} mean {statistics.fmean([x[0] for x in hot[h:]]):.1f}")
tot_hot=sum(x[0] for x in [r for r in []] ) 
