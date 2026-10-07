"""Decode step time at long context: per context, the prompt (real source code,
as ../2026-10-07-decode-mem-dive/longctx.py) is prefilled and cached first,
then decoded with agentic sampling; step time = the server's decode seconds /
draft steps over that request (vllm:request_decode_time / spec drafts).

    decode_long.py --tokens 50000,130000 --out rows.jsonl
"""
import argparse
import json
import sys
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
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = json.loads(urllib.request.urlopen(f"{BASE}/v1/models").read())["data"][0]["id"]
    files = sorted(Path("vllm").rglob("*.py"))
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}" for f in files)
    out = open(a.out, "a")
    for n in map(int, a.tokens.split(",")):
        chars = n * 3
        while True:
            text = corpus[:chars]
            got = call("/tokenize", {"model": name, "prompt": text})["count"]
            if abs(got - n) < 0.02 * n:
                break
            chars = int(chars * n / got)
        msgs = [{"role": "user", "content": "Below is source code from a project.\n" + text
                 + "\n\nExplain in detail how the code above is organised and how a "
                 "request flows through it, citing concrete functions and files."}]
        call("/v1/chat/completions", {"model": name, "messages": msgs, "max_tokens": 1})
        for seed in range(a.seeds):
            m0 = AB.scrape(BASE)
            r = call("/v1/chat/completions", {"model": name, "messages": msgs,
                     "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 0.95,
                     "seed": seed})
            m1 = AB.scrape(BASE)
            d = {k: m1[k] - m0[k] for k in m0}
            row = {"ctx": r["usage"]["prompt_tokens"], "out": r["usage"]["completion_tokens"],
                   "seed": seed, "step_ms": round(1000 * d["decode_s"] / d["steps"], 3),
                   "tokens_per_step": round(1 + d["accepted"] / d["steps"], 3),
                   "steps": d["steps"]}
            print(json.dumps(row), flush=True)
            out.write(json.dumps(row) + "\n")


main()
