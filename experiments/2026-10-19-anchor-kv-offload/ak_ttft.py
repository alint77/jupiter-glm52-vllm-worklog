import json, statistics as st
R="/e/fscratch/profound/naeimitabiei1/sliced-prof/"
def ev(t):
    out={}
    for l in open(f"{R}{t}/events.jsonl"):
        e=json.loads(l)
        if e.get("err") is None and "first" in e: out[(e["agent"],e["index"])]=(e["first"]-e["send"], e["input"], e["end"]-e["first"], e["output"])
    return out
for hold, arms in [("2412782","bc"),("2413890","b"),("2413891","bc"),("2413893","b")]:
    A=ev(f"aka-tlnsys-{hold}")
    for x in arms:
        B=ev(f"ak{x}-tlnsys-{hold}"); k=sorted(set(A)&set(B))
        big=[B[i][0]/A[i][0] for i in k if A[i][0]>5]
        small=[B[i][0]-A[i][0] for i in k if A[i][0]<=5]
        dec=[(B[i][2]/max(B[i][3]-1,1))/(A[i][2]/max(A[i][3]-1,1)) for i in k if A[i][3]>100 and B[i][3]==A[i][3]]
        print(f"{hold} {x}: matched {len(k)}  big-prefill TTFT {x}/a median {st.median(big):.3f} (n={len(big)})  "
              f"small TTFT {x}-a median {st.median(small)*1e3:+.0f} ms (n={len(small)})  decode time/token {x}/a {st.median(dec) if dec else float('nan'):.3f} (n={len(dec)})")
