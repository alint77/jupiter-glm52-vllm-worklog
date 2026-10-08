"""Draft acceptance per prompt and domain (prompts.json, 16 prompts: 8 code,
3 math, 3 prose, 2 other), one request at a time so the server's
spec-decode counters (totals and per draft position) belong to that request.

Two modes, each with --seeds samples at temperature 1.0 / top_p 0.95:
  content   the chat prompt rendered by the model's own template with the
            think block closed (<think></think>), sent to /v1/completions:
            the model writes the answer itself (code, maths, prose)
  thinking  /v1/chat/completions as served: reasoning first

    ood_accept.py --out acc.jsonl [--seeds 2] [--max-tokens 512]
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8027"
MODEL_DIR = "/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-W4A16"


def post(path, body):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def counters():
    text = urllib.request.urlopen(f"{BASE}/metrics", timeout=60).read().decode()
    out = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "pos": {}}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        val = float(line.split()[-1])
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] += val
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["draft_tokens"] += val
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] += val
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            m = re.search(r'position="(\d+)"', line)
            if m:
                p = int(m.group(1))
                out["pos"][p] = out["pos"].get(p, 0.0) + val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--modes", default="content,thinking")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    name = json.loads(urllib.request.urlopen(f"{BASE}/v1/models").read())["data"][0]["id"]
    prompts = json.loads((HERE / "prompts.json").read_text())
    with open(a.out, "w") as f:
        for mode in a.modes.split(","):
            for p in prompts:
                for seed in range(a.seeds):
                    c0 = counters()
                    if mode == "content":
                        text = tok.apply_chat_template(
                            [{"role": "user", "content": p["prompt"]}],
                            add_generation_prompt=True, tokenize=False) + "</think>"
                        r = post("/v1/completions", {
                            "model": name, "prompt": text, "max_tokens": a.max_tokens,
                            "temperature": 1.0, "top_p": 0.95, "seed": seed})
                        out_text = r["choices"][0]["text"]
                    else:
                        r = post("/v1/chat/completions", {
                            "model": name, "max_tokens": a.max_tokens, "temperature": 1.0,
                            "top_p": 0.95, "seed": seed,
                            "messages": [{"role": "user", "content": p["prompt"]}]})
                        msg = r["choices"][0]["message"]
                        out_text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + \
                            (msg.get("content") or "")
                    c1 = counters()
                    drafts = c1["drafts"] - c0["drafts"]
                    acc = c1["accepted"] - c0["accepted"]
                    pos = {k: (c1["pos"].get(k, 0) - c0["pos"].get(k, 0)) / max(drafts, 1)
                           for k in sorted(c1["pos"])}
                    row = {"id": p["id"], "domain": p["domain"], "mode": mode, "seed": seed,
                           "tokens": r["usage"]["completion_tokens"], "drafts": drafts,
                           "accepted": acc, "acc_len": 1 + acc / max(drafts, 1),
                           "pos_rate": pos, "head": out_text[:200]}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"{mode:8s} {p['id']:14s} seed {seed}: acc_len {row['acc_len']:.2f} "
                          f"tokens {row['tokens']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
