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

# ---- #2 grid / M-tile re-read factor ----
print("=== Marlin launch geometry (rank0, chunk 3) ===")
ops=list(chunks(sorted(ROOT.glob("*_rank0.*"))[0]))[2]
seen={}
for k in ops:
    if "marlin_moe_wna16" not in k["name"] or not k.get("op"): continue
    d=k["op"]["args"]["Input Dims"]; E=d[2][0]; gemm="w13" if d[0][1]==6144 else "w2"
    g=tuple(k["args"].get("grid")); b=tuple(k["args"].get("block"))
    key=(gemm,E)
    if key in seen: continue
    seen[key]=(g,b,k["dur"],k["args"].get("shared memory"),k["args"].get("est. achieved occupancy %"))
for key in sorted(seen, key=lambda x:(x[0],-x[1]))[:10]:
    g,b,dur,sm,occ=seen[key]
    print(f"  {key[0]:4s} E={key[1]:2d}  grid={g} block={b} smem={sm} occ={occ}%  {dur:8.0f}us")

# ---- #4 prefill all-reduce parity, all ranks ----
print("\n=== prefill all-reduce: even/odd ordinal split, cross-rank ===")
per={}
for p in sorted(ROOT.glob("*.trace.json.gz")):
    r=int(p.name.split("_rank")[1].split(".")[0]); steps=[]
    for ops in chunks(p):
        ar=[k["dur"] for k in ops if "ncclDevKernel_AllReduce" in k["name"]]
        if len(ar)==160: steps.append(ar)
    per[r]=steps
n=min(len(v) for v in per.values())
od=collections.defaultdict(list)
for s in range(n):
    for o in range(160): od[o].append(per[0][s][o])
ev_=[o for o in od if o%2==0]; odd=[o for o in od if o%2==1]
for nm,idx in (("even",ev_),("odd",odd)):
    meds=[statistics.median(od[o]) for o in idx]
    print(f"  rank0 {nm} ordinals: n={len(idx)} mean-of-medians {statistics.fmean(meds):8.1f} us  "
          f"sum {sum(meds)/1000:7.2f} ms/chunk")
mins=[];means=[]
for s in range(n):
    for o in range(160):
        d=[per[r][s][o] for r in sorted(per)]
        mins.append(min(d)); means.append(statistics.fmean(d))
print(f"  cross-rank: sum-of-minima {sum(mins)/n/1000:.1f} ms  sum-of-means {sum(means)/n/1000:.1f} ms")
ev_min=sum(min(per[r][s][o] for r in per) for s in range(n) for o in range(0,160,2))/n/1000
od_min=sum(min(per[r][s][o] for r in per) for s in range(n) for o in range(1,160,2))/n/1000
ev_mean=sum(statistics.fmean([per[r][s][o] for r in per]) for s in range(n) for o in range(0,160,2))/n/1000
od_mean=sum(statistics.fmean([per[r][s][o] for r in per]) for s in range(n) for o in range(1,160,2))/n/1000
print(f"  even: transfer {ev_min:6.2f} ms  total {ev_mean:6.2f} ms  -> wait {ev_mean-ev_min:6.2f}")
print(f"  odd : transfer {od_min:6.2f} ms  total {od_mean:6.2f} ms  -> wait {od_mean-od_min:6.2f}")
