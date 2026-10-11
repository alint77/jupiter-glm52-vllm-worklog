"""Per-arm (k) means over nodes and reps from sliced-prof/il-k<k>-<job>/bench.jsonl
and gsm8k.json."""
import collections
import glob
import json
import os
import statistics as st

root = f"/e/fscratch/profound/{os.environ['USER']}/sliced-prof"
by = collections.defaultdict(lambda: collections.defaultdict(list))
gsm = collections.defaultdict(list)
for d in sorted(glob.glob(f"{root}/il-k*-*")):
    k = d.split("/il-k")[1].split("-")[0]
    p = f"{d}/bench.jsonl"
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            w = r["window"]
            if w == "mixed":
                by[k]["mixed prefill TTFT s"].append(r["prefill_ttft"])
                by[k]["mixed decoder max gap s"].append(max(x["max_gap_s"] for x in r["decoders"]))
                by[k]["mixed decoder tok/s"].append(st.mean(x["decode_tps"] for x in r["decoders"]))
            else:
                by[k][f"{w} makespan s"].append(r["makespan"])
                by[k][f"{w} mean TTFT s"].append(st.mean(x["ttft"] for x in r["reqs"]))
                by[k][f"{w} max TTFT s"].append(max(x["ttft"] for x in r["reqs"]))
                by[k][f"{w} decode tok/s"].append(st.mean(x["decode_tps"] for x in r["reqs"]))
    g = f"{d}/gsm8k.json"
    if os.path.exists(g):
        gsm[k].append(json.load(open(g)).get("accuracy"))
ks = sorted(by)
metrics = list(dict.fromkeys(m for k in ks for m in by[k]))
print("| metric | " + " | ".join(f"k={k}" for k in ks) + " |")
print("| --- |" + " ---: |" * len(ks))
for m in metrics:
    cells = []
    for k in ks:
        v = by[k][m]
        cells.append(f"{st.mean(v):.2f} ({len(v)})" if v else "-")
    print(f"| {m} | " + " | ".join(cells) + " |")
print("| GSM8K | " + " | ".join(" / ".join(f"{a:.3f}" for a in gsm[k]) for k in ks) + " |")
