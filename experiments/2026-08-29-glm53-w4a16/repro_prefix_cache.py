#!/usr/bin/env python3
"""Minimal probe: does prefix caching change greedy output?

At temperature 0 the same prompt must produce the same completion regardless of
what is cached. This sends a long shared prefix with several suffixes, then
re-sends the first one, and diffs. Any difference between the first and the
repeat is a cache-correctness bug, not sampling noise.

Also records whether the *first* request through a cold cache differs from the
same request once the prefix is resident, which is the pattern GSM8K showed.
"""

import argparse
import json
import urllib.request
from difflib import unified_diff


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8027")
    p.add_argument("--model", default="glm52-w4a16-tiered")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--prefix-tokens", type=int, default=900)
    p.add_argument("--output", default="-")
    return p.parse_args()


def complete(args, prompt: str) -> dict:
    body = json.dumps({
        "model": args.model, "prompt": prompt,
        "max_tokens": args.max_tokens, "temperature": 0.0, "seed": 0,
    }).encode()
    req = urllib.request.Request(
        f"{args.base_url}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return {"text": d["choices"][0]["text"],
            "finish": d["choices"][0].get("finish_reason"),
            "tokens": d["usage"]["completion_tokens"],
            "cached": (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")}


def main() -> None:
    args = parse_args()
    # A long shared prefix, like the 5-shot GSM8K preamble.
    shared = ("The following are grade school math problems with worked "
              "solutions.\n\n") + "".join(
        f"Question: A shop sells {i} apples per crate and has {i * 3} crates. "
        f"How many apples?\nAnswer: {i} * {i * 3} = {i * i * 3}. "
        f"The answer is {i * i * 3}.\n\n" for i in range(2, 40))
    suffixes = [
        "Question: A train travels 60 km in 1.5 hours. What is its average "
        "speed in km/h?\nAnswer:",
        "Question: Sam has 24 marbles and gives away a third. How many are "
        "left?\nAnswer:",
        "Question: A book costs 12 dollars. How much do 7 books cost?\nAnswer:",
    ]

    results = []
    # Pass 1: cold cache -- first request populates the shared prefix.
    for i, s in enumerate(suffixes):
        results.append(("cold", i, complete(args, shared + s)))
    # Pass 2: warm cache -- identical prompts, prefix now resident.
    for i, s in enumerate(suffixes):
        results.append(("warm", i, complete(args, shared + s)))

    report = {"mismatches": [], "detail": []}
    for i in range(len(suffixes)):
        cold = next(r for k, j, r in results if k == "cold" and j == i)
        warm = next(r for k, j, r in results if k == "warm" and j == i)
        same = cold["text"] == warm["text"]
        report["detail"].append({
            "suffix": i, "identical": same,
            "cold_tokens": cold["tokens"], "warm_tokens": warm["tokens"],
            "cold_cached": cold["cached"], "warm_cached": warm["cached"],
        })
        if not same:
            report["mismatches"].append({
                "suffix": i,
                "diff": list(unified_diff(
                    cold["text"].splitlines(), warm["text"].splitlines(),
                    fromfile="cold", tofile="warm", lineterm=""))[:40],
            })
    report["verdict"] = (
        "PREFIX CACHE CHANGES GREEDY OUTPUT" if report["mismatches"]
        else "identical across cold and warm cache")
    text = json.dumps(report, indent=2)
    if args.output == "-":
        print(text)
    else:
        open(args.output, "w").write(text + "\n")
        print(report["verdict"], f"({len(report['mismatches'])} mismatched)")


if __name__ == "__main__":
    main()
