"""Score a short GSM8K smoke and save the generated text.

The qualification smoke refuses a zero-accuracy run, which is right for a gate
but useless for attributing one. This variant always writes its result, keeps
the completions, and never exits nonzero, so a ladder of arms can be compared.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "2026-09-05-cold-prefetch"),
)

from gsm8k_paired import INVALID, _build_gsm8k_prompts, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-questions", type=int, default=8)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--port", type=int, default=8129)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts, labels = _build_gsm8k_prompts(args.num_questions, args.num_shots, "")
    start = time.perf_counter()
    preds, texts = asyncio.run(
        run(prompts, labels, f"http://127.0.0.1:{args.port}", args.max_tokens,
            args.concurrency)
    )
    elapsed = time.perf_counter() - start
    correct = [int(p == label) for p, label in zip(preds, labels)]
    accuracy = sum(correct) / len(correct)
    invalid = sum(1 for p in preds if p == INVALID) / len(preds)
    Path(args.out).write_text(
        json.dumps(
            {
                "arm": args.arm,
                "accuracy": accuracy,
                "invalid": invalid,
                "num_questions": len(correct),
                "seconds": elapsed,
                "correct": correct,
                "preds": preds,
                "labels": labels,
                "texts": texts,
            },
            indent=2,
        )
    )
    print(
        f"[{args.arm}] accuracy {accuracy:.4f} invalid {invalid:.3f} "
        f"on {len(correct)} questions in {elapsed:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
