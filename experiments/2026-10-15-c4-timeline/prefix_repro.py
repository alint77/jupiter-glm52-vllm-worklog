"""Prefix-cache hits of consecutive replayed turns at c=1: sends each listed
cc-replay index in order (non-streamed, max_tokens 16) and records the
server's prefix-cache query/hit token deltas and TTFT-ish wall per request.
The rendered prompts of #113->#114, #16->#17, #63->#64 are append-only
(prefix_diverge.py); #160->#161 changes the tool list (expected miss).

    prefix_repro.py --out rows.jsonl [--seq 112,113,114 16,17 ...]
"""
import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8027"
REPLAY = Path("/e/fscratch/profound/naeimitabiei1/agentic-bench/cc-replay.jsonl")
SEQS = ("112,113,114", "16,17", "63,64", "160,161")


def metrics():
    text = urllib.request.urlopen(f"{BASE}/metrics", timeout=30).read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^vllm:(prefix_cache_\w+_total|kv_cache_usage_perc)"
                     r"(?:\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", nargs="*",
                    default=os.environ.get("PREFIX_SEQS", "__".join(SEQS)).split("__"))
    a = ap.parse_args()
    want = {int(i) for s in a.seq for i in s.split(",")}
    rows = {}
    for line in REPLAY.open():
        r = json.loads(line)
        if r["index"] in want:
            rows[r["index"]] = r
    model = json.loads(urllib.request.urlopen(f"{BASE}/v1/models").read())["data"][0]["id"]
    with open(a.out, "w") as f:
        for s in a.seq:
            for i in (int(x) for x in s.split(",")):
                body = dict(rows[i]["body"], model=model, stream=False, max_tokens=16)
                req = urllib.request.Request(
                    f"{BASE}/v1/messages", data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", "x-api-key": "none",
                             "anthropic-version": "2023-06-01"})
                m0, t0 = metrics(), time.time()
                reply = json.loads(urllib.request.urlopen(req, timeout=1800).read())
                wall, m1 = time.time() - t0, metrics()
                q = m1["prefix_cache_queries_total"] - m0["prefix_cache_queries_total"]
                h = m1["prefix_cache_hits_total"] - m0["prefix_cache_hits_total"]
                row = {"seq": s, "index": i, "input": reply["usage"]["input_tokens"],
                       "queried": q, "hit": h, "hit_pct": round(100 * h / q, 1) if q else None,
                       "wall_s": round(wall, 2), "kv_after": m1["kv_cache_usage_perc"]}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
