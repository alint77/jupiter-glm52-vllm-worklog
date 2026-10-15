"""c=4 agentic timeline: 4 closed-loop agents replay consecutive Claude Code
requests (cc-replay.jsonl, the long session) from staggered offsets, streamed
on /v1/messages so each request logs send / first token / end. A sampler
thread logs the server's scheduler gauges and node memory/CPU every 0.5 s, and
two Nsight Systems windows (WINDOWS) are taken mid-run.

    timeline_run.py --out timings.jsonl   (events.jsonl, samples.jsonl beside)

TIMELINE_DURATION=<s> changes the run length; TIMELINE_NOPROF=1 takes no
profiler windows (A/B runs).
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8027"
REPLAY = Path("/e/fscratch/profound/naeimitabiei1/agentic-bench/cc-replay.jsonl")
SESSION = "898193f4"
OFFSETS = (0, 48, 96, 144)
DURATION = float(os.environ.get("TIMELINE_DURATION", 270))
WINDOWS = () if os.environ.get("TIMELINE_NOPROF") else ((60.0, 6.0), (170.0, 6.0))
MAX_TOKENS = 2048
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting",
          "vllm:kv_cache_usage_perc", "vllm:prompt_tokens_total",
          "vllm:generation_tokens_total", "vllm:prefix_cache_hits_total",
          "vllm:prefix_cache_queries_total")

lock = threading.Lock()
stop = threading.Event()


def emit(f, row):
    with lock:
        f.write(json.dumps(row) + "\n")
        f.flush()


def get(path, timeout=30, post=False):
    req = urllib.request.Request(f"{BASE}{path}", data=b"" if post else None)
    return urllib.request.urlopen(req, timeout=timeout).read()


def agent(i, reqs, model, f, t0):
    for k, r in enumerate(reqs):
        if stop.is_set():
            return
        body = dict(r["body"], model=model, stream=True,
                    max_tokens=min(r["body"]["max_tokens"], MAX_TOKENS))
        req = urllib.request.Request(
            f"{BASE}/v1/messages", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-api-key": "none",
                     "anthropic-version": "2023-06-01"})
        send = time.time()
        first = None
        usage = {}
        nbytes = len(req.data)
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                for line in resp:
                    if not line.startswith(b"data:"):
                        continue
                    ev = json.loads(line[5:])
                    t = ev.get("type")
                    if t == "content_block_delta" and first is None:
                        first = time.time()
                    elif t == "message_start":
                        usage.update(ev["message"].get("usage", {}))
                    elif t == "message_delta":
                        usage.update(ev.get("usage", {}))
                    if stop.is_set() and time.time() - send > 600:
                        break
            err = None
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:300]
        end = time.time()
        emit(f, {"agent": i, "k": k, "index": r["index"], "send": send - t0,
                 "first": (first - t0) if first else None, "end": end - t0,
                 "body_kb": nbytes // 1024, "input": usage.get("input_tokens"),
                 "output": usage.get("output_tokens"), "err": err})
        print(f"a{i} #{r['index']} in={usage.get('input_tokens')} "
              f"out={usage.get('output_tokens')} ttft="
              f"{(first - send) if first else None} e2e={end - send:.1f}", flush=True)


def sampler(f, t0):
    pids = None
    while not stop.is_set():
        t = time.time()
        row = {"t": t - t0}
        try:
            text = get("/metrics", 5).decode()
            for line in text.splitlines():
                m = re.match(r"^([a-z_:]+)(?:\{[^}]*\})? ([0-9.eE+-]+)$", line)
                if m and m.group(1) in GAUGES:
                    row[m.group(1)[5:]] = row.get(m.group(1)[5:], 0) + float(m.group(2))
        except Exception:  # noqa: BLE001
            pass
        mi = dict(l.split(":", 1) for l in open("/proc/meminfo"))
        row["host_used_gib"] = (int(mi["MemTotal"].split()[0])
                                - int(mi["MemAvailable"].split()[0])) / 2**20
        row["load1"] = float(open("/proc/loadavg").read().split()[0])
        try:
            ps = subprocess.run(["ps", "-u", str(__import__("os").getuid()), "-o",
                                 "pcpu=,rss=,comm="], capture_output=True,
                                text=True, timeout=5).stdout
            cpu = {}
            for l in ps.splitlines():
                p = l.split(None, 2)
                if len(p) == 3:
                    cpu[p[2][:20]] = cpu.get(p[2][:20], 0) + float(p[0])
            row["cpu"] = {k: round(v, 1) for k, v in cpu.items() if v > 5}
        except Exception:  # noqa: BLE001
            pass
        emit(f, row)
        time.sleep(max(0.0, 0.5 - (time.time() - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    model = json.loads(get("/v1/models"))["data"][0]["id"]
    rows = [json.loads(l) for l in REPLAY.open()]
    sess = [r for r in rows if r["session"].startswith(SESSION)]
    chains = [sess[o:o + 48] for o in OFFSETS]
    ev = open(out.parent / "events.jsonl", "w")
    sm = open(out.parent / "samples.jsonl", "w")
    tm = open(out, "w")
    t0 = time.time()
    emit(tm, {"t0_unix": t0, "offsets": OFFSETS, "windows": WINDOWS})
    threads = [threading.Thread(target=sampler, args=(sm, t0), daemon=True)]
    threads += [threading.Thread(target=agent, args=(i, c, model, ev, t0), daemon=True)
                for i, c in enumerate(chains)]
    for th in threads:
        th.start()
    for w, (at, length) in enumerate(WINDOWS, 1):
        time.sleep(max(0.0, at - (time.time() - t0)))
        s = time.time() - t0
        get("/start_profile", 60, post=True)
        time.sleep(length)
        e = time.time() - t0
        get("/stop_profile", 120, post=True)
        emit(tm, {"window": w, "start": s, "stop_called": e,
                  "stopped": time.time() - t0})
        print(f"window {w}: {s:.1f}-{e:.1f}", flush=True)
    time.sleep(max(0.0, DURATION - (time.time() - t0)))
    stop.set()
    time.sleep(2)
    emit(tm, {"end": time.time() - t0})


if __name__ == "__main__":
    main()
