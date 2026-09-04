import collections, gzip, json
from pathlib import Path
GPU={"kernel","gpu_memcpy","gpu_memset"}; LAU={"cuda_runtime","cuda_driver"}
bad=0; tot=0; sums=collections.Counter()
for p in sorted(Path("/e/project1/profound/alint77/traces/mtp3-profile-1665068/prefill").glob("*.trace.json.gz")):
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
    for s,en in [(a["t"],b["t"]) for a,b in zip(ann,ann[1:])]:
        corr=[e["args"]["correlation"] for e in lau if s<=e["t"]<en and "correlation" in e.get("args",{})]
        ops=sorted((k for c in corr for k in bc[c]),key=lambda e:e["t"])
        rows=[]
        for k in ops:
            if "marlin_moe_wna16" not in k["name"]: continue
            l=[q for q in cpu.get(k["args"].get("External id"),[]) if q["cat"]=="cpu_op"]
            if not l: continue
            d=min(l,key=lambda q:q["dur"])["args"]["Input Dims"]
            a,bq,sc=d[0],d[2],d[4]
            by=bq[0]*bq[1]*bq[2]*4+sc[0]*sc[1]*sc[2]*2
            rows.append(("w13" if a[1]==6144 else "w2",bq[0],k["dur"],by/(k["dur"]*1e-6)/1e9))
        for gemm in ("w13","w2"):
            r=[x for x in rows if x[0]==gemm]
            for i in range(0,len(r)-1,2):
                a,b=r[i],r[i+1]; tot+=1
                sums[a[1]+b[1]]+=1
                if not (a[3] > b[3]): bad+=1
print(f"pairs checked {tot}, ordering violations (first not faster) {bad} = {100*bad/tot:.2f}%")
print("sum-of-experts per pair:", dict(sums))
