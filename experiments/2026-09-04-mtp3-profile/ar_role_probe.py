import collections, gzip, json, statistics
from pathlib import Path
GPU={"kernel","gpu_memcpy","gpu_memset"}; LAU={"cuda_runtime","cuda_driver"}
ROOT=Path("/e/project1/profound/alint77/traces/mtp3-profile-1665068/prefill")
def chunks(p):
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
        cs=[e["args"]["correlation"] for e in lau if s<=e["t"]<en and "correlation" in e.get("args",{})]
        ops=sorted((k for c in cs for k in bc[c]),key=lambda e:e["t"])
        if not ops: continue
        for k in ops:
            l=[q for q in cpu.get(k["args"].get("External id"),[]) if q["cat"]=="cpu_op"]
            k["op"]=min(l,key=lambda q:q["dur"]) if l else None
        yield ops
def classify(ops):
    """Label each AR by the last significant kernel before it."""
    out=[]
    for i,k in enumerate(ops):
        if "ncclDevKernel_AllReduce" not in k["name"]: continue
        role="?"
        for j in range(i-1,max(0,i-9),-1):
            nm=ops[j]["name"]
            if "nvjet" in nm or "marlin" in nm or "moe_sum" in nm or "add_" in nm.lower():
                if "nvjet" in nm: role="post-attention (o_proj)"
                elif "moe_sum" in nm or "marlin" in nm: role="post-MoE"
                else: role="post-MoE"
                break
            if "elementwise" in nm and "add" in nm: role="post-MoE"; break
        out.append((role,k["dur"],i))
    return out
per=collections.defaultdict(list); nch=0
allr={}
for p in sorted(ROOT.glob("*.trace.json.gz")):
    r=int(p.name.split("_rank")[1].split(".")[0]); allr[r]=[]
    for ops in chunks(p):
        c=classify(ops); allr[r].append(c)
        if r==0:
            nch+=1
            for role,dur,_ in c: per[role].append(dur)
print(f"rank0, {nch} chunks: all-reduce by role")
for role,v in sorted(per.items(), key=lambda i:-sum(i[1])):
    print(f"  {role:26s} n/chunk {len(v)/nch:5.1f}  median {statistics.median(v):7.1f} us  "
          f"sum {sum(v)/nch/1000:7.2f} ms/chunk")
n=min(len(v) for v in allr.values())
tr=collections.defaultdict(float); tot=collections.defaultdict(float)
for s in range(n):
    m=min(len(allr[r][s]) for r in allr)
    for o in range(m):
        role=allr[0][s][o][0]
        ds=[allr[r][s][o][1] for r in sorted(allr)]
        tr[role]+=min(ds)/n/1000; tot[role]+=statistics.fmean(ds)/n/1000
print("\ncross-rank split (ms/chunk):")
for role in sorted(tot, key=lambda k:-tot[k]):
    print(f"  {role:26s} transfer {tr[role]:6.2f}  total {tot[role]:6.2f}  wait {tot[role]-tr[role]:6.2f}"
          f"  ({100*(tot[role]-tr[role])/tot[role]:4.1f}% wait)")
