#!/usr/bin/env python3
"""Score GSM8K answers, and check that speculative decoding is actually lossless.

Everything measured in this phase so far has been acceptance and throughput.
Nothing has checked whether the model gets the answers right, and "coherent
output" is not a correctness gate.

Two things are tested here, and the second is the sharper one:

  1. **Accuracy.** Standard GSM8K scoring, the last integer in the completion
     against the ground truth after `####`, matching tests/evals/gsm8k.

  2. **Losslessness.** DFlash2's card states that greedy output matches the
     target exactly. So at temperature 0, a speculator must produce
     byte-identical completions to no speculation at all. Any divergence is a
     correctness bug in the draft path, not a speed question -- and it would
     not show up in acceptance numbers, which is exactly how a broken drafter
     could hide.

Run once per speculator against the same prompts, then diff the saved
completions across runs.
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8027"
INVALID = -9999999


def answer_value(text: str) -> int:
    """Last integer in the text, the convention tests/evals/gsm8k uses."""
    numbers = re.findall(r"-?\d+", text.replace(",", ""))
    if not numbers:
        return INVALID
    try:
        return int(numbers[-1])
    except ValueError:
        return INVALID


def label_value(answer: str) -> int:
    tail = answer.split("####")[-1]
    return answer_value(tail)


def chat(prompt: str, model: str, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 13,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="glm52-w4a16-tiered")
    ap.add_argument(
        "--dataset",
        default="/e/project1/profound/alint77/models/datasets/gsm8k/test.jsonl",
    )
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    items = []
    with open(args.dataset) as f:
        for line in f:
            if len(items) >= args.num_samples:
                break
            items.append(json.loads(line))

    chat(items[0]["question"], args.model, 32)  # warm

    records, correct, invalid = [], 0, 0
    for i, it in enumerate(items):
        text = chat(it["question"], args.model, args.max_tokens)
        pred, gold = answer_value(text), label_value(it["answer"])
        ok = pred == gold
        correct += ok
        invalid += pred == INVALID
        records.append(
            {"i": i, "pred": pred, "gold": gold, "correct": ok, "completion": text}
        )
        if (i + 1) % 16 == 0:
            print(f"  {i + 1:3d}/{len(items)}  accuracy {correct / (i + 1):.3f}", flush=True)

    result = {
        "label": args.label,
        "n": len(records),
        "accuracy": round(correct / len(records), 4),
        "invalid_rate": round(invalid / len(records), 4),
        "mean_completion_chars": round(
            sum(len(r["completion"]) for r in records) / len(records), 1
        ),
        "temperature": 0,
        "note": "greedy; a lossless speculator must match no-speculation exactly",
        "records": records,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    summary = {k: v for k, v in result.items() if k != "records"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
