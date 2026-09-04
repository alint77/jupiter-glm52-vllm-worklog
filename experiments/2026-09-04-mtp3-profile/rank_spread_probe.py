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
        if ops: yield ops
# per-layer: marlin time and MLA time, per rank
mar=collections.defaultdict(list); mla=collections.defaultdict(list)
for p in sorted(ROOT.glob("*.trace.json.gz")):
    r=int(p.name.split("_rank")[1].split(".")[0])
    for ci,ops in enumerate(chunks(p)):
        m=[k["dur"] for k in ops if "marlin_moe_wna16" in k["name"]]
        a=[k["dur"] for k in ops if "flash_fwd_splitkv_mla" in k["name"]]
        mar[r].append(m); mla[r].append(a)
n=min(len(mar[r]) for r in mar)
def spread(store, per_layer):
    tot_mean=0.0; tot_max=0.0
    for s in range(n):
        L=min(len(store[r][s]) for r in store)
        for i in range(0, L-per_layer+1, per_layer):
            vals=[sum(store[r][s][i:i+per_layer]) for r in sorted(store)]
            tot_mean+=statistics.fmean(vals); tot_max+=max(vals)
    return tot_mean/n/1000, tot_max/n/1000
for nm,store,pl in (("routed MoE Marlin (4/layer)",mar,4),("sparse MLA (1/layer)",mla,1)):
    mean,mx=spread(store,pl)
    print(f"{nm:30s} mean-rank {mean:7.2f} ms  slowest-rank {mx:7.2f} ms  "
          f"per-layer excess {mx-mean:6.2f} ms/chunk ({100*(mx-mean)/mean:4.1f}%)")
