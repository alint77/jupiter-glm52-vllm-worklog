#!/usr/bin/env python3
"""Capture what the no-speculator path actually emits on a five-shot prompt.

The vllm GSM8K eval reports aggregates and discards completions, so the
no-speculator failure -- 20 output tokens per question and 78% unparseable --
could not be diagnosed from its output. This sends the same five-shot prompt
three ways so the failing axis is unambiguous:

  A. max_tokens 256 with the eval's stop sequences   (the failing mode)
  B. max_tokens 256, no stop sequences
  C. max_tokens 2048 with stop sequences

If A returns near-empty text and B does not, the fault is stop-sequence
handling. If A and B are both empty but C is fine, it is length-related. If
all three are fine, the earlier run was not reproducible and the failure lay
elsewhere.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8027"
GSM8K = "/e/project1/profound/alint77/models/datasets/gsm8k/train.jsonl"
TEST = "/e/project1/profound/alint77/models/datasets/gsm8k/test.jsonl"


def build_five_shot(n_shot: int = 5) -> tuple[str, str]:
    """Same construction the vllm eval uses: Question/Answer pairs, then a query."""
    shots = []
    with open(GSM8K) as f:
        for line in f:
            if len(shots) >= n_shot:
                break
            d = json.loads(line)
            shots.append(f"Question: {d['question']}\nAnswer: {d['answer']}")
    with open(TEST) as f:
        q = json.loads(f.readline())["question"]
    return "\n\n".join(shots) + f"\n\nQuestion: {q}\nAnswer:", q


def completion(prompt: str, max_tokens: int, stop) -> dict:
    body = {
        "model": "glm52-w4a16-tiered",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if stop:
        body["stop"] = stop
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt, question = build_five_shot()
    print(f"five-shot prompt: {len(prompt)} chars, query: {question[:70]}...\n")

    cases = [
        ("A max256 +stop", 256, ["Question:", "\n\n"]),
        ("B max256 -stop", 256, None),
        ("C max2048 +stop", 2048, ["Question:", "\n\n"]),
    ]
    out = {"prompt_chars": len(prompt), "cases": []}
    for name, mt, stop in cases:
        r = completion(prompt, mt, stop)
        ch = r["choices"][0]
        text = ch.get("text", "")
        usage = r.get("usage", {})
        rec = {
            "case": name,
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": ch.get("finish_reason"),
            "stop_reason": ch.get("stop_reason"),
            "text_len": len(text),
            "text": text[:600],
        }
        out["cases"].append(rec)
        print(f"=== {name} ===")
        print(f"  tokens={rec['completion_tokens']}  finish={rec['finish_reason']}  "
              f"stop_reason={rec['stop_reason']}  chars={rec['text_len']}")
        print(f"  {text[:300]!r}\n")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
