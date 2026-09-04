#!/usr/bin/env python3
"""Can a cold-expert H2D prefetch be fully hidden behind layer compute?

Design under test: a double-buffered staging area in HBM holds the cold experts
for the upcoming layer. The moment layer L's cold Marlin finishes, its buffer is
free and the H2D for layer L+1 starts; it must land before layer L+1's cold
Marlin begins.

The window is therefore measured, not assumed: end of layer L's last cold Marlin
kernel to the start of layer L+1's first cold Marlin kernel. Bytes are the exact
operand sizes of layer L+1's two cold Marlin calls.
"""
from __future__ import annotations
import collections, gzip, json, statistics, sys
from pathlib import Path

H2D = 450e9          # NVLink-C2C per direction, JSC spec
H2D_MEAS = 373e9     # measured achievable Grace read
EXPERT_MIB = 20.3
GPU={"kernel","gpu_memcpy","gpu_memset"}; LAU={"cuda_runtime","cuda_driver"}
ROOT = Path("/e/project1/profound/alint77/traces/mtp3-profile-1665068/prefill")

def chunks(p):
    with gzip.open(p,"rt") as fh: ev=[e for e in json.load(fh)["traceEvents"] if e.get("ph")=="X"]
    o=min(e["ts"] for e in ev if e.get("cat") in GPU)
    for e in ev: e["t"]=(e["ts"]-o)/1000
    cpu=collections.defaultdict(list)
    for e in ev:
        if e.get("cat")=="cpu_op": cpu[e["args"].get("External id")].append(e)
    bc=collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in GPU and e.get("args",{}).get("correlation") is not None:
            bc[e["args"]["correlation"]].append(e)
    lau=sorted((e for e in ev if e.get("cat") in LAU),key=lambda e:e["t"])
    ann=sorted((e for e in ev if e.get("cat")=="user_annotation" and e["name"].startswith("execute_")),key=lambda e:e["t"])
    for s,en in [(a["t"],b["t"]) for a,b in zip(ann,ann[1:])]:
        cs=[e["args"]["correlation"] for e in lau if s<=e["t"]<en and "correlation" in e.get("args",{})]
        ops=sorted((k for c in cs for k in bc[c]),key=lambda e:e["t"])
        if not ops: continue
        for k in ops:
            l=[q for q in cpu.get(k["args"].get("External id"),[]) if q["cat"]=="cpu_op"]
            k["op"]=min(l,key=lambda q:q["dur"]) if l else None
        yield ops, en-s

def layers(ops):
    """Per layer: cold bytes, cold start/end, hot and cold ms.

    Walks Marlin calls with a resync: a layer normally contributes four calls
    (hot w13, hot w2, cold w13, cold w2), but a layer whose 64 experts all sit
    in one tier contributes two. Grouping blindly by four would drift from
    there on, so each candidate group is validated on hot+cold == 64 and a
    failure advances by two instead.
    """
    mar=[k for k in ops if "marlin_moe_wna16" in k["name"] and k.get("op")]
    def dims(k): return k["op"]["args"]["Input Dims"]
    def wb(k):
        d=dims(k); bq,sc=d[2],d[4]
        return bq[0]*bq[1]*bq[2]*4+sc[0]*sc[1]*sc[2]*2
    out=[]; i=0; single=0
    while i+3 < len(mar):
        g=mar[i:i+4]
        w13=[k for k in g if dims(k)[0][1]==6144]
        w2=[k for k in g if dims(k)[0][1]!=6144]
        if len(w13)==2 and len(w2)==2 and dims(w13[0])[2][0]+dims(w13[1])[2][0]==64:
            hot_w13,cold_w13=w13; hot_w2,cold_w2=w2
            cold=[cold_w13,cold_w2]
            out.append({"bytes":wb(cold_w13)+wb(cold_w2),
                        "E":dims(cold_w13)[2][0],
                        "start":min(k["t"] for k in cold),
                        "end":max(k["t"]+k["dur"]/1000 for k in cold),
                        "hot_ms":(hot_w13["dur"]+hot_w2["dur"])/1000,
                        "cold_ms":(cold_w13["dur"]+cold_w2["dur"])/1000,
                        "single":False})
            i+=4
        else:
            single+=1
            i+=2
    return out, single

p=sorted(ROOT.glob("*_rank0.*trace.json.gz"))[0]
allc=list(chunks(p))
rows=[]
singles=0
for ops,wall in allc:
    L,sg=layers(ops); singles+=sg
    for a,b in zip(L,L[1:]):
        window=b["start"]-a["end"]          # buffer free -> next cold Marlin starts
        need=b["bytes"]/H2D*1000
        need_m=b["bytes"]/H2D_MEAS*1000
        rows.append({"window":window,"need":need,"need_m":need_m,"bytes":b["bytes"],
                     "E":b["E"],"hot":b["hot_ms"],"cold":b["cold_ms"]})
n=len(allc)
print(f"rank0, {n} chunks, {len(rows)} layer transitions, "
      f"{singles} single-tier (all-64) groups resynced\n")
w=sorted(r["window"] for r in rows); nd=sorted(r["need"] for r in rows)
print(f"prefetch window (end of cold L -> start of cold L+1):")
print(f"   min {w[0]:6.2f}  p5 {w[len(w)//20]:6.2f}  median {w[len(w)//2]:6.2f}  max {w[-1]:6.2f} ms")
print(f"H2D time needed for next layer's cold experts @450 GB/s:")
print(f"   min {nd[0]:6.2f}  median {nd[len(nd)//2]:6.2f}  max {nd[-1]:6.2f} ms")
slack=sorted(r["window"]-r["need"] for r in rows)
ratio=sorted(r["window"]/r["need"] for r in rows)
print(f"\nslack (window - need): min {slack[0]:6.2f} ms   median {slack[len(slack)//2]:6.2f} ms")
print(f"window/need ratio:     min {ratio[0]:6.2f}x  median {ratio[len(ratio)//2]:6.2f}x")
bad=[r for r in rows if r["window"]<r["need"]]
badm=[r for r in rows if r["window"]<r["need_m"]]
print(f"transitions that CANNOT hide @450 GB/s: {len(bad)}/{len(rows)}")
print(f"transitions that CANNOT hide @373 GB/s measured: {len(badm)}/{len(rows)}")

per_chunk_bytes=sum(r["bytes"] for r in rows)/n
print(f"\ncold bytes per chunk {per_chunk_bytes/1e9:6.2f} GB -> "
      f"{per_chunk_bytes/H2D*1000:6.1f} ms @450, {per_chunk_bytes/H2D_MEAS*1000:6.1f} ms @373")
busy=statistics.fmean([w for _,w in allc])   # measured wall per chunk
print(f"duty cycle against a {busy:.0f} ms chunk: "
      f"{100*per_chunk_bytes/H2D*1000/busy:.1f}% @450")

mx=max(r["bytes"] for r in rows)
print(f"\nbuffer sizing: largest layer {mx/2**20:6.0f} MiB ({max(r['E'] for r in rows)} experts); "
      f"double-buffered {2*mx/2**30:.2f} GiB")
hot_per_expert=statistics.fmean(r["hot"] for r in rows)/31.0
proj=statistics.fmean(r["E"] for r in rows)*hot_per_expert*75
cur=statistics.fmean(r["cold"] for r in rows)*75
print(f"\nif cold then runs at hot's HBM rate ({hot_per_expert*1000:.0f} us/expert):")
print(f"   cold Marlin {cur:6.1f} ms -> {proj:6.1f} ms   saving {cur-proj:6.1f} ms/chunk "
      f"({100*(cur-proj)/1964.6:.1f}% of the chunk)")
