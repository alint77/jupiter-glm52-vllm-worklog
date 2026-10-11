"""Profiled interleave windows (../2026-10-11-interleave/bench_interleave.py's
mixed and fanout cases), one Nsight Systems report each:

- mixed:  3 decoders at ~10K context stream 1500 tokens; 4 s in, a cached
          60K prefix + 8K new tokens arrives (1 output token). Window: from
          0.5 s before that arrival to 0.5 s after its first token.
- fanout: 4 prompts at once sharing a cached 20K prefix plus 4K new each,
          300 tokens each. Window: the first 3 s.

    prof_mixed.py --out timings.jsonl
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-11-interleave"))
from bench_interleave import prompt, run, summary  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
from conc_probe import call  # noqa: E402


def window(start_s, stop_s):
    """Profile from start_s to stop_s after now, on a thread."""
    def go():
        t0 = time.time()
        time.sleep(start_s)
        call("/start_profile").read()
        time.sleep(max(0.0, t0 + stop_s - time.time()))
        call("/stop_profile").read()
    th = threading.Thread(target=go)
    th.start()
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    mb = 1_000_000
    warm = lambda text: run(name, [(0, text, 1)])  # noqa: E731
    base60 = prompt(name, 60000, corpus)
    base20 = prompt(name, 20000, corpus[int(1.5 * mb):])
    warm(base60)
    warm(base20)
    rows = []
    off = 2 * mb
    ctx = [prompt(name, 10000, corpus[off + i * mb // 2:]) for i in range(3)]
    for t in ctx:
        warm(t)
    delta = prompt(name, 8000, corpus[off + 2 * mb:])
    # un-profiled first, for the prefill TTFT without profiler overhead
    for prof in (False, True):
        th = window(3.5, 6.2) if prof else None
        outs = run(name, [(0, t, 1500) for t in ctx] + [(4, base60 + delta, 1)])
        if th:
            th.join()
        rows.append({"window": "mixed", "profiled": prof, "prefill_ttft": outs[-1]["ttft"],
                     "decoders": [summary(o) for o in outs[:-1]]})
        print(json.dumps(rows[-1]), flush=True)
        delta = prompt(name, 8000, corpus[off + 3 * mb:])
        time.sleep(3)
    for prof, k in ((False, 6), (True, 10)):
        news = [prompt(name, 4000, corpus[off + (k + i) * mb // 2:]) for i in range(4)]
        th = window(0.0, 3.0) if prof else None
        outs = run(name, [(0, base20 + t, 300) for t in news])
        if th:
            th.join()
        t0 = min(o["t0"] for o in outs)
        rows.append({"window": "fanout", "profiled": prof,
                     "makespan": max(o["end"] for o in outs) - t0,
                     "reqs": [summary(o) for o in outs]})
        print(json.dumps(rows[-1]), flush=True)
        time.sleep(3)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
