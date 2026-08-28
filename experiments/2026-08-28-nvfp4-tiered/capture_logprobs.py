#!/usr/bin/env python3
"""Capture teacher-forced next-token distributions for a fixed prompt set.

Used to test whether the tiered MoE path changes what the model computes.
`prompt_logprobs` scores every position of a supplied prompt, so both arms
evaluate identical token sequences and nothing can diverge the way sampled
generations do -- comparing generations would only re-measure floating-point
non-determinism, which we already know differs across batch shapes.

Writes, per position, the top-k token ids and their logprobs. Compare two
captures with compare_logprobs.py.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Fixed, varied, and deliberately not from any training-adjacent set. Short
# enough that a full capture is cheap, long enough to exercise many experts.
PROMPTS = [
    "The capital of France is Paris, and the capital of Germany is",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:",
    "In 1969, Apollo 11 landed on the Moon. The commander of that mission was",
    "The derivative of x^3 + 2x with respect to x is",
    "Translate to French: 'The weather is cold today.' ->",
    "A train travels 60 miles in 1.5 hours. Its average speed in miles per hour is",
    "class LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, value):",
    "The three laws of thermodynamics describe energy, entropy, and",
]


def completion(port: int, model: str, prompt: str, topk: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": topk,
            "logprobs": topk,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8027)
    ap.add_argument("--model", default="glm52-w4a16-tiered")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    captures = []
    for i, p in enumerate(PROMPTS):
        resp = completion(args.port, args.model, p, args.topk)
        ch = resp["choices"][0]
        plp = ch.get("prompt_logprobs") or []
        positions = []
        for pos in plp:
            if not pos:
                continue  # first token has no distribution
            # {token_id: {"logprob": x, "rank": r, "decoded_token": s}}
            items = sorted(
                ((int(k), v["logprob"]) for k, v in pos.items()),
                key=lambda kv: -kv[1],
            )[: args.topk]
            positions.append({"ids": [k for k, _ in items], "lps": [v for _, v in items]})
        captures.append({"prompt_index": i, "num_positions": len(positions), "positions": positions})
        print(f"  prompt {i}: {len(positions)} scored positions", flush=True)

    out = {
        "label": args.label,
        "model": args.model,
        "topk": args.topk,
        "num_prompts": len(PROMPTS),
        "total_positions": sum(c["num_positions"] for c in captures),
        "captures": captures,
    }
    Path(args.out).write_text(json.dumps(out))
    print(f"\nwrote {args.out}: {out['total_positions']} positions across {len(PROMPTS)} prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
