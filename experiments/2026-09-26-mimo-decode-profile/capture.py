#!/usr/bin/env python3
"""Profile MiMo-V2.6 decode steps on a running server, with an unprofiled control.

Reuses the GLM harness's engine-state gating (2026-09-04-mtp3-profile): a
window opens only once generation tokens are rising, so it cannot land in
prefill. For each context length, one long request is started; its step time
is first measured unprofiled from the metrics counters, then a torch-profiler
window is opened on the *same* request, so the distortion estimate compares
like with like.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-04-mtp3-profile"))
import capture as harness  # noqa: E402


CHAT = False


def completion(prompt: str, max_tokens: int, temperature: float) -> None:
    """Greedy (0) or the model's own sampling defaults (top_p 0.95).

    With CHAT set, the prompt goes to /v1/chat/completions as one user message:
    the routed-expert trace writer (VLLM_ROUTING_TRACE_DIR) is only wired into
    the chat endpoint.
    """
    body = {"model": harness.MODEL, "max_tokens": max_tokens,
            "temperature": temperature, "ignore_eos": True}
    if temperature:
        body["top_p"] = 0.95
    if CHAT:
        body["messages"] = [{"role": "user", "content": prompt}]
        harness.post("/v1/chat/completions", body)
    else:
        body["prompt"] = prompt
        harness.post("/v1/completions", body)


def counters() -> dict[str, float]:
    values = harness.metrics()
    return {
        "gen": values.get("generation_tokens_total", 0.0),
        "drafts": values.get("spec_decode_num_drafts_total", 0.0),
        "accepted": values.get("spec_decode_num_accepted_tokens_total", 0.0),
        "t": time.time(),
    }


def rate(before: dict, after: dict) -> dict:
    steps = after["drafts"] - before["drafts"]
    gen = after["gen"] - before["gen"]
    seconds = after["t"] - before["t"]
    return {
        "seconds": round(seconds, 3),
        "steps": steps,
        "tokens": gen,
        "step_ms": round(seconds * 1000 / steps, 3) if steps else None,
        "tokens_per_step": round(gen / steps, 3) if steps else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-prompts", type=Path, required=True)
    parser.add_argument("--short-prompts", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--window", type=float, default=2.0)
    # 0 reproduces the first capture; greedy decoding looped on the 96K prompt
    # (every draft accepted), so later captures sample like real traffic.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--chat", action="store_true",
                        help="use the chat endpoint (needed for routed-expert traces)")
    args = parser.parse_args()
    global CHAT
    CHAT = args.chat
    harness.MODEL = "mimo26-pro"
    long_prompt = json.loads(args.long_prompts.read_text().splitlines()[1])["prompt"]
    short_prompt = json.loads(args.short_prompts.read_text().splitlines()[0])["prompt"]
    report = {}
    for label, prompt in (("decode-short", short_prompt), ("decode-96k", long_prompt)):
        print(f"--- {label} ---", flush=True)
        thread = threading.Thread(
            target=completion, args=(prompt, 6000, args.temperature), daemon=True
        )
        thread.start()
        harness.wait_for_decode(1, timeout=900)
        time.sleep(1.0)
        before = counters()
        time.sleep(4.0)
        unprofiled = rate(before, counters())
        harness.post("/start_profile")
        before = counters()
        time.sleep(args.window)
        # Read before stopping: stop_profile blocks while traces serialize.
        profiled = rate(before, counters())
        try:
            harness.post("/stop_profile")
        except Exception as error:  # workers still flush their traces
            print(f"stop_profile failed, continuing: {error}", flush=True)
        thread.join()
        time.sleep(45)
        files = harness.move_traces(args.trace_root, label)
        report[label] = {"unprofiled": unprofiled, "profiled_window": profiled,
                         "trace_files": files}
        print(json.dumps(report[label]), flush=True)
    (args.trace_root / "capture-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
