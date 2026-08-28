#!/usr/bin/env python3
"""Acceptance length of a running speculative server, from its Prometheus counters.

Acceptance length is the figure the Phase 42 decision rule is stated in:
accepted-plus-bonus tokens per verification step. It is what decides DFlash2
against MTP3 on this target, and it is independent of expert residency, so it
can be measured on a trimmed placement profile.

Reports the delta across a fixed generation, not the process totals, so a
warmup cannot contaminate it.
"""

import argparse
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8027"

# vLLM's spec-decode counters. num_drafts counts verification steps;
# num_accepted_tokens counts draft tokens that survived. Every step also emits
# one bonus token, which is what makes acceptance length accepted/drafts + 1.
COUNTERS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"


def get(path: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
        return r.read().decode()


def scrape() -> dict:
    out: dict = {"per_pos": {}}
    for line in get("/metrics").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        base = name.split("{", 1)[0]
        if base in {f"{c}_total" for c in COUNTERS} or base in COUNTERS:
            out[base.removesuffix("_total")] = out.get(
                base.removesuffix("_total"), 0.0
            ) + float(value)
        elif base.startswith(PER_POS):
            pos = ""
            if "position=" in name:
                pos = name.split('position="', 1)[1].split('"', 1)[0]
            out["per_pos"][pos] = out["per_pos"].get(pos, 0.0) + float(value)
    return out


def delta(before: dict, after: dict) -> dict:
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in COUNTERS}
    d["per_pos"] = {
        k: after["per_pos"].get(k, 0.0) - before["per_pos"].get(k, 0.0)
        for k in after["per_pos"]
    }
    return d


def generate(prompt: str, max_tokens: int, model: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 13,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["text"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="glm52-w4a16-tiered")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--warmup-tokens", type=int, default=64)
    ap.add_argument(
        "--prompt",
        default=(
            "Write a Python function that merges two sorted lists into one "
            "sorted list without using sorted(). Explain the invariant it "
            "maintains, then give the complexity."
        ),
    )
    args = ap.parse_args()

    generate(args.prompt, args.warmup_tokens, args.model)
    time.sleep(1.0)

    before = scrape()
    t0 = time.perf_counter()
    text = generate(args.prompt, args.max_tokens, args.model)
    wall = time.perf_counter() - t0
    after = scrape()
    d = delta(before, after)

    drafts = d["vllm:spec_decode_num_drafts"]
    draft_tokens = d["vllm:spec_decode_num_draft_tokens"]
    accepted = d["vllm:spec_decode_num_accepted_tokens"]

    result = {
        "label": args.label,
        "wall_s": round(wall, 4),
        "verification_steps": drafts,
        "draft_tokens": draft_tokens,
        "accepted_draft_tokens": accepted,
        # One bonus token per verification step, always accepted.
        "acceptance_length": round((accepted + drafts) / drafts, 4) if drafts else None,
        "draft_acceptance_rate": round(accepted / draft_tokens, 4)
        if draft_tokens
        else None,
        "step_time_ms": round(1000.0 * wall / drafts, 4) if drafts else None,
        "per_position_accepted": d["per_pos"],
        "text_head": text[:200],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    a = result["acceptance_length"]
    print(json.dumps({k: v for k, v in result.items() if k != "text_head"}, indent=2))
    if a is not None:
        # Phase 42 decision rule, fixed before the run.
        verdict = (
            "REFUTED (< 3.35: cannot beat MTP3 under any step-time assumption)"
            if a < 3.35
            else "AMBIGUOUS (3.35-4.19: measured step time decides)"
            if a < 4.19
            else "CLEARS (> 4.19: beats MTP3 under both bounds)"
        )
        print(f"\nacceptance length {a} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
