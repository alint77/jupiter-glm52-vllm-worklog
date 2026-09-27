#!/usr/bin/env python3
"""Decode step time on fixed prompts, greedy, for a same-content A/B.

Greedy speculative decoding is lossless, so every arm generates the same
tokens and routes the same experts: step time differs only by the config.
Per request, step time is measured from the engine counters between the
first decode step and the last poll before the request finishes.

    bench.py --out result.json [--reps 3]
"""

import argparse
import importlib.util
import json
import statistics
import threading
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "glm_capture", Path(__file__).resolve().parent / "capture.py")
_glm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_glm)
harness, mimo, prompt_for = _glm.harness, _glm.mimo, _glm.prompt_for


def one(prompt: str, max_tokens: int) -> dict:
    thread = threading.Thread(target=mimo.completion, args=(prompt, max_tokens, 0.0),
                              daemon=True)
    thread.start()
    harness.wait_for_decode(1, timeout=1500)
    first = last = mimo.counters()
    while thread.is_alive():
        sample = mimo.counters()
        if harness.metrics().get("num_requests_running", 0) >= 1:
            last = sample
        time.sleep(0.25)
    thread.join()
    return mimo.rate(first, last)


def ttft(prompt: str) -> float:
    """Wall time of a one-token request: prefill plus one sampling step."""
    start = time.perf_counter()
    mimo.completion(prompt, 1, 0.0)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--contexts", nargs="+", default=["short", "96k"])
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--ttft-reps", type=int, default=0)  # decode is the focus
    args = parser.parse_args()
    harness.MODEL = "glm53-w4a16-tiered"
    one(prompt_for("short"), 64)  # warm-up
    result = {}
    for context in args.contexts if args.reps else ():
        prompt = prompt_for(context)
        runs = [one(prompt, args.max_tokens) for _ in range(args.reps)]
        result[context] = {
            "runs": runs,
            "step_ms_median": statistics.median(r["step_ms"] for r in runs),
            "tokens_per_step": runs[0]["tokens_per_step"],
        }
        print(context, json.dumps(result[context]), flush=True)
    for context in args.contexts if args.ttft_reps else ():
        prompt = prompt_for(context)
        times = [ttft(prompt) for _ in range(args.ttft_reps if context == "short" else 2)]
        if context == "short":  # a pause between samples, so each starts idle
            times = []
            for _ in range(args.ttft_reps):
                times.append(ttft(prompt))
                time.sleep(0.5)
        result[f"ttft_{context}"] = {"runs": times, "median_s": statistics.median(times)}
        print(f"ttft_{context}", json.dumps(result[f"ttft_{context}"]), flush=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
