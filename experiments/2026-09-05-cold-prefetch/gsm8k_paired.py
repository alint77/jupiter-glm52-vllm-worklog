# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One GSM8K pass that records per-question outcomes, for paired comparison.

`gsm8k_eval.py` saves only aggregate accuracy, which cannot separate a real
effect from this configuration's run-to-run noise: phase 3b measured a 0.6-2.3
nat spread on repeated identical requests, baseline included. Comparing two
independent accuracies at a few hundred questions has a confidence interval on
the *difference* far wider than any defect worth catching.

Recording each question's outcome allows a paired comparison on the identical
question set, and running two passes per arm gives the within-arm discordance
that the cross-arm discordance has to be judged against.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/evals/gsm8k"))

import aiohttp  # noqa: E402
from gsm8k_eval import (  # noqa: E402
    INVALID,
    _build_gsm8k_prompts,
    call_vllm_api,
    get_answer_value,
)


async def run(prompts, labels, url, max_tokens, concurrency):
    preds: list[int] = [0] * len(prompts)
    texts: list[str] = [""] * len(prompts)
    gate = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:

        async def one(index: int) -> None:
            async with gate:
                answer, _ = await call_vllm_api(
                    session=session,
                    prompt=prompts[index],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    stop=["Question", "Assistant:", "<|separator|>"],
                    url=url,
                )
            texts[index] = answer
            preds[index] = get_answer_value(answer)

        await asyncio.gather(*(one(i) for i in range(len(prompts))))
    return preds, texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-questions", type=int, default=400)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--port", type=int, default=8027)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts, labels = _build_gsm8k_prompts(args.num_questions, args.num_shots, "")
    # call_vllm_api appends /v1/completions itself; give it the base.
    url = f"http://127.0.0.1:{args.port}"
    start = time.perf_counter()
    preds, texts = asyncio.run(
        run(prompts, labels, url, args.max_tokens, args.concurrency)
    )
    elapsed = time.perf_counter() - start
    correct = [int(p == label) for p, label in zip(preds, labels)]
    accuracy = sum(correct) / len(correct)
    Path(args.out).write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "num_questions": len(correct),
                "seconds": elapsed,
                "correct": correct,
                "preds": preds,
                "labels": labels,
            },
            indent=2,
        )
    )
    invalid = sum(1 for p in preds if p == INVALID) / len(preds)
    print(
        f"accuracy {accuracy:.4f} on {len(correct)} questions in {elapsed:.0f}s "
        f"({len(correct) / elapsed:.2f} q/s), invalid {invalid:.3f}"
    )
    # A broken transport returns empty strings fast and scores zero, which is
    # indistinguishable from a real result in the saved file. Refuse to hand
    # back a run that plainly never reached the model.
    if invalid > 0.5 or accuracy == 0.0:
        print("FAIL: the run did not produce usable answers")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
