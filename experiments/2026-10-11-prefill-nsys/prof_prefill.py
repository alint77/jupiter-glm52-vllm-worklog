"""Prefill windows for Nsight Systems (one report per /start_profile..
/stop_profile range under NSYS_OUT), each first timed twice un-profiled on
fresh text (outside any capture range):

- fresh16k:     16K new tokens, empty cache (four 4,096-token chunks, context 0-16K)
- delta8k@60k:  8K new tokens after a cached 60K prefix (two chunks at 60-68K):
                the shape of a Claude Code turn after an Explore agent reads files
- mixed:        delta8k@60k while 3 other requests decode at ~10K context

    prof_prefill.py --out timings.jsonl
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
from conc_probe import call, prompt, stream  # noqa: E402

HEAD = "Below is source code from a project.\n"
TAIL = "\n\nExplain in detail how the code above is organised."


def ttft(name, text):
    t0 = time.time()
    r = json.loads(call("/v1/chat/completions", {"model": name, "max_tokens": 1, "messages": [
        {"role": "user", "content": HEAD + text + TAIL}]}).read())
    return time.time() - t0, r["usage"]["prompt_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    mb = 1_000_000
    base = prompt(name, 60000, corpus[0:])
    ttft(name, base)  # cache the 60K prefix
    fresh = [prompt(name, 16000, corpus[(1 + i) * mb:]) for i in range(3)]
    delta = [prompt(name, 8000, corpus[(4 + i) * mb:]) for i in range(6)]
    ctx = [prompt(name, 10000, corpus[(10 + i) * mb:]) for i in range(3)]
    for t in ctx:
        ttft(name, t)
    rows = []

    def window(label, text, profiled, decoders=False):
        threads, outs = [], []
        if decoders:
            for i, t in enumerate(ctx):
                o = {}
                th = threading.Thread(target=stream, args=(name, t, i, o, 3000))
                th.start()
                threads.append(th)
                outs.append(o)
            time.sleep(4)
        if profiled:
            call("/start_profile").read()
        s, n = ttft(name, text)
        if profiled:
            time.sleep(0.3)
            call("/stop_profile").read()
            time.sleep(3)
        for th in threads:
            th.join()
        row = {"label": label, "profiled": profiled, "ttft_s": s, "prompt_tokens": n}
        if outs:
            row["decoders"] = [{k: o.get(k) for k in ("decode_tps", "tokens")} for o in outs]
        rows.append(row)
        print(json.dumps(row), flush=True)

    for i in range(2):
        window("fresh16k", fresh[i], False)
        window("delta8k@60k", base + delta[i], False)
    window("mixed", base + delta[2], False, decoders=True)
    window("fresh16k", fresh[2], True)
    window("delta8k@60k", base + delta[3], True)
    window("mixed", base + delta[4], True, decoders=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
