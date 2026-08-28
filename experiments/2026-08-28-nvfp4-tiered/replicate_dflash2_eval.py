#!/usr/bin/env python3
"""Replicate the DFlash2 card's acceptance-length protocol against our server.

The card reports 5.94 on GSM8K for DFlash2 and 5.12 for MTP, both at seven
draft tokens per verification step. Our 16K-code harness gives 3.323 for
DFlash2 at the same width, 41.5% of ceiling against their 74.2%. This script
removes the harness as an explanation by matching their protocol point for
point:

  - GSM8K questions, short prompts, not 16K of packed source
  - the model's chat template, so the completion is a real assistant turn
  - temperature 1.0, top_p 0.95, the card's stated sampling
  - natural EOS, no forced length: our suite ran --ignore-eos to 512 tokens,
    which makes the model generate past its own stopping point, and a drafter
    cannot predict text the target itself would not have written
  - up to 4096 new tokens

Acceptance length is defined as the card defines it: completion tokens divided
by verification steps, which for our counters is
(accepted_draft_tokens + verification_steps) / verification_steps.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8027"


def get(path: str, timeout: float = 60.0) -> str:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
        return r.read().decode()


def spec_counters() -> tuple[float, float]:
    drafts = accepted = 0.0
    for line in get("/metrics").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        base = name.split("{", 1)[0]
        if base.startswith("vllm:spec_decode_num_drafts"):
            drafts += float(value)
        elif base.startswith("vllm:spec_decode_num_accepted_tokens") and "per_pos" not in base:
            accepted += float(value)
    return drafts, accepted


def chat(prompt: str, model: str, max_tokens: int, temperature: float, top_p: float) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="glm52-w4a16-tiered")
    ap.add_argument(
        "--dataset",
        default="/e/project1/profound/alint77/models/datasets/gsm8k/test.jsonl",
    )
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    questions = []
    with open(args.dataset) as f:
        for line in f:
            if len(questions) >= args.num_samples:
                break
            questions.append(json.loads(line)["question"])

    # Warm the server so compilation and graph capture are outside the window.
    chat(questions[0], args.model, 64, args.temperature, args.top_p)

    per_request = []
    t0 = time.perf_counter()
    for i, q in enumerate(questions):
        d0, a0 = spec_counters()
        resp = chat(q, args.model, args.max_tokens, args.temperature, args.top_p)
        d1, a1 = spec_counters()
        drafts, accepted = d1 - d0, a1 - a0
        completion = resp.get("usage", {}).get("completion_tokens", 0)
        if drafts > 0:
            per_request.append(
                {
                    "acceptance_length": (accepted + drafts) / drafts,
                    "completion_tokens": completion,
                    "verification_steps": drafts,
                    "accepted": accepted,
                }
            )
        if (i + 1) % 16 == 0:
            mean = statistics.mean(r["acceptance_length"] for r in per_request)
            print(f"  {i + 1:4d}/{len(questions)}  running mean acceptance {mean:.3f}", flush=True)
    wall = time.perf_counter() - t0

    lengths = [r["acceptance_length"] for r in per_request]
    total_steps = sum(r["verification_steps"] for r in per_request)
    total_acc = sum(r["accepted"] for r in per_request)
    result = {
        "label": args.label,
        "requests": len(per_request),
        "wall_s": round(wall, 2),
        # The card's definition: per-request mean.
        "acceptance_length_mean": round(statistics.mean(lengths), 4) if lengths else None,
        "acceptance_length_median": round(statistics.median(lengths), 4) if lengths else None,
        # Pooled, for comparison with our other harness.
        "acceptance_length_pooled": round((total_acc + total_steps) / total_steps, 4)
        if total_steps
        else None,
        "mean_completion_tokens": round(
            statistics.mean(r["completion_tokens"] for r in per_request), 1
        )
        if per_request
        else None,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "protocol": "DFlash2 card: chat template, natural EOS, T=1.0 top_p=0.95",
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("\ncard reference, GSM8K at 7 draft tokens: DFlash2 5.94, MTP 5.12")
    return 0


if __name__ == "__main__":
    sys.exit(main())
