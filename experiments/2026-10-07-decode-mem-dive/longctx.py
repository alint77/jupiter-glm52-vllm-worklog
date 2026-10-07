"""Long-context decode probe: a chat request over N tokens of real source code
(this repo's vllm/*.py, deterministic order), agentic sampling (temperature
1.0, top_p 0.95), one optional torch-profiler window per request opened after
30 decode steps (agentic_bench.Profiler). Contexts grow as extensions of the
same prefix, so later requests hit the prefix cache.

    longctx.py --tokens 50000,130000 [--trace-root DIR --profile]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-28-agentic-decode-bench"))
import agentic_bench as AB  # noqa: E402

BASE = "http://127.0.0.1:8027"


def call(path, body):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="50000,130000")
    ap.add_argument("--trace-root", type=Path)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1200)
    a = ap.parse_args()
    name = json.loads(urllib.request.urlopen(f"{BASE}/v1/models").read())["data"][0]["id"]
    files = sorted(Path("vllm").rglob("*.py"))
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}" for f in files)
    prof = AB.Profiler(BASE, a.trace_root, 99 if a.profile else 0, 3.0, 40)
    prof.done = 10  # window-10.. : after the agentic windows in the same root
    for n in map(int, a.tokens.split(",")):
        chars = n * 3
        while True:
            text = corpus[:chars]
            got = call("/tokenize", {"model": name, "prompt": text})["count"]
            if abs(got - n) < 0.02 * n:
                break
            chars = int(chars * n / got)
        msgs = [{"role": "user", "content":
                 "Below is source code from a project.\n" + text +
                 "\n\nExplain in detail how the code above is organised and how a "
                 "request flows through it, citing concrete functions and files."}]
        body = {"model": name, "messages": msgs, "max_tokens": a.max_tokens,
                "temperature": 1.0, "top_p": 0.95, "seed": 0}
        t = time.time()
        r = prof.around(lambda: call("/v1/chat/completions", body))
        u = r["usage"]
        print(f"ctx {u['prompt_tokens']} out {u['completion_tokens']} "
              f"{time.time() - t:.1f} s profiled={prof.last}", flush=True)


main()
