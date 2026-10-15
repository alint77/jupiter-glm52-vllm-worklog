"""Request- and node-level view of a timeline_run.py run.

Per request (events.jsonl): TTFT, decode time and rate. Per 10 s bin
(samples.jsonl): running / waiting, prompt and generation tok/s, KV use,
prefix hit rate, host memory, CPU % per process.

    reqs.py <run dir>
"""
import json
import statistics as st
import sys
from pathlib import Path

d = Path(sys.argv[1])
ev = [json.loads(l) for l in open(d / "events.jsonl")]
sm = [json.loads(l) for l in open(d / "samples.jsonl")]
tm = [json.loads(l) for l in open(d / "timings.jsonl")]
print("windows:", [r for r in tm if "window" in r])

print("\nper request (s): agent k index | send ttft decode | in out | out tok/s")
rows = []
for r in sorted(ev, key=lambda r: r["send"]):
    ttft = r["first"] - r["send"] if r["first"] else None
    dec = r["end"] - r["first"] if r["first"] else None
    rate = (r["output"] or 0) / dec if dec else None
    rows.append((ttft, dec, rate, r))
    print(f"  a{r['agent']} {r['k']:2d} #{r['index']:3d} | {r['send']:6.1f} "
          f"{ttft if ttft is not None else -1:6.2f} {dec if dec is not None else -1:6.2f} | "
          f"{r['input']} {r['output']} | {rate or 0:6.1f} {r['err'] or ''}")
ok = [x for x in rows if x[0] is not None]
if ok:
    print(f"\nrequests {len(ok)}: TTFT median {st.median(x[0] for x in ok):.2f} s, "
          f"max {max(x[0] for x in ok):.2f}; decode tok/s median "
          f"{st.median(x[2] for x in ok if x[2]):.1f}")

print("\n10 s bins: t | run wait | prompt/s gen/s | kv% hit% | host GiB load | cpu%")
prev = None
for b in range(0, int(sm[-1]["t"]) + 1, 10):
    w = [s for s in sm if b <= s["t"] < b + 10 and "num_requests_running" in s]
    if not w:
        continue
    last = w[-1]
    if prev is None:
        prev = w[0]
    dt = last["t"] - prev["t"] or 1
    pt = (last["prompt_tokens_total"] - prev["prompt_tokens_total"]) / dt
    gt = (last["generation_tokens_total"] - prev["generation_tokens_total"]) / dt
    q = last["prefix_cache_queries_total"] - prev["prefix_cache_queries_total"]
    h = last["prefix_cache_hits_total"] - prev["prefix_cache_hits_total"]
    cpu = {}
    for s in w:
        for k, v in s.get("cpu", {}).items():
            cpu[k] = cpu.get(k, 0) + v / len(w)
    print(f"  {b:4d} | {st.mean(s['num_requests_running'] for s in w):3.1f} "
          f"{st.mean(s['num_requests_waiting'] for s in w):3.1f} | {pt:7.0f} {gt:5.1f} | "
          f"{100 * st.mean(s['kv_cache_usage_perc'] for s in w):4.1f} "
          f"{100 * h / q if q else 0:4.0f} | {last['host_used_gib']:6.1f} {last['load1']:5.1f} | "
          + " ".join(f"{k[:14]}={v:.0f}" for k, v in sorted(cpu.items())))
    prev = last
