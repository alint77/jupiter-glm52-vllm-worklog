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
s,en=[(a["t"],b["t"]) for a,b in zip(ann,ann[1:])][2]
corr=[e["args"]["correlation"] for e in lau if s<=e["t"]<en and "correlation" in e.get("args",{})]
ops=sorted((k for c in corr for k in bc[c]),key=lambda e:e["t"])
seq=[]
for k in ops:
    if "marlin_moe_wna16" not in k["name"]: continue
    l=[q for q in cpu.get(k["args"].get("External id"),[]) if q["cat"]=="cpu_op"]
    if not l: continue
    d=min(l,key=lambda q:q["dur"])["args"]["Input Dims"]
    a,bq,sc=d[0],d[2],d[4]
    by=bq[0]*bq[1]*bq[2]*4+sc[0]*sc[1]*sc[2]*2
    seq.append({"gemm":"w13" if a[1]==6144 else "w2","E":bq[0],"us":k["dur"],
                "by":by,"bw":by/(k["us"] if False else k["dur"]*1e-6)/1e9,"t":k["t"]})
print(f"{len(seq)} marlin calls in chunk 3")
big=[x for x in seq if x["E"]==64]
print(f"\nall-64 calls: {len(big)}")
for x in big:
    print(f"  {x['gemm']:4s} E=64  {x['us']:8.0f} us  {x['by']/2**20:7.0f} MiB  {x['bw']:6.0f} GB/s  t={x['t']:.0f}ms")
# reference rates
for gemm in ("w13","w2"):
    r=[x for x in seq if x["gemm"]==gemm and x["E"]!=64]
    pairs=[(r[i],r[i+1]) for i in range(0,len(r)-1,2) if r[i]["E"]+r[i+1]["E"]==64]
    hot=[a["bw"] for a,b in pairs]; cold=[b["bw"] for a,b in pairs]
    print(f"  reference {gemm}: hot median {statistics.median(hot):5.0f} GB/s, "
          f"cold median {statistics.median(cold):5.0f} GB/s")
print("\n=> an all-64 call at ~hot rate means hot=64/cold=0 (nothing to prefetch);")
print("   at ~cold rate it means cold=64 and the slot must be sized for 64 experts.")
# what does the placement profile say for the extreme layers?
d=json.load(open("agent_space/profiles/glm53-w4a16-2496.json"))
hotc=[]
for L in range(len(d["hot_experts"])):
    hs=set(d["hot_experts"][L]); ow=d["owners"][L]
    hotc.append(sum(1 for e in hs if ow[e]==0))
print(f"\nprofile hot-per-layer for rank0: min {min(hotc)} max {max(hotc)}; "
      f"layers with 0 hot: {sum(1 for x in hotc if x==0)}, with 64 hot: {sum(1 for x in hotc if x==64)}")
