#!/usr/bin/env python3
"""Capture per-question GSM8K completions so two servers can be diffed.

MTP is lossless by construction: at temperature 0 a drafted token is accepted
only when it equals the target's own argmax. So an MTP server and a
no-speculator server must emit *identical* token sequences for identical
prompts. Any divergence is a correctness bug, and the first divergent position
localises it.

Writes {"prompts": [...], "outputs": [...], "token_ids": [...]} for offline
comparison.
"""

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8027")
    p.add_argument("--model", default="glm52-w4a16-tiered")
    p.add_argument("--eval-dir", default="tests/evals/gsm8k")
    p.add_argument("--num-questions", type=int, default=64)
    p.add_argument("--num-shots", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.eval_dir)
    import gsm8k_eval

    prompts, labels = gsm8k_eval._build_gsm8k_prompts(
        num_questions=args.num_questions, num_shots=args.num_shots)

    def one(prompt: str) -> dict:
        body = json.dumps({
            "model": args.model, "prompt": prompt,
            "max_tokens": args.max_tokens, "temperature": 0.0,
            "logprobs": 0,
        }).encode()
        req = urllib.request.Request(
            f"{args.base_url}/v1/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1200) as r:
            d = json.loads(r.read())
        ch = d["choices"][0]
        lp = ch.get("logprobs") or {}
        return {"text": ch["text"], "tokens": lp.get("tokens"),
                "finish": ch.get("finish_reason")}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, prompts))

    json.dump({"prompts": prompts, "labels": labels, "results": results},
              open(args.output, "w"))
    print(f"captured {len(results)} completions -> {args.output}")


if __name__ == "__main__":
    main()
