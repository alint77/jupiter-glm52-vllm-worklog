#!/usr/bin/env python3
"""Profile GLM-5.3 decode steps at several context lengths, each with an
unprofiled control on the same request.

The MiMo harness (../2026-09-26-mimo-decode-profile/capture.py) with GLM's
served name and a third, long context: the 96K Claude-Code-shaped prompts of
../2026-09-04-mtp3-profile concatenated, to see what grows with context
(the DSA indexer scans every cached token; sparse MLA reads a fixed 2048).

    capture.py --trace-root DIR [--contexts short 96k 290k] [--temperature 1]
"""

import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    # both harnesses are also called capture.py, so load them by path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


harness = _load("capture", HERE.parent / "2026-09-04-mtp3-profile/capture.py")
mimo = _load("mimo_capture", HERE.parent / "2026-09-26-mimo-decode-profile/capture.py")
LONG = HERE.parent / "2026-09-04-mtp3-profile/prompts.jsonl"
SHORT = HERE.parent / "2026-08-29-glm53-routing-capture/prompts-short.jsonl"


def prompt_for(context: str) -> str:
    long = [json.loads(line)["prompt"] for line in LONG.read_text().splitlines()]
    if context == "short":
        return json.loads(SHORT.read_text().splitlines()[0])["prompt"]
    if context == "96k":
        return long[1]
    if context == "290k":   # three ~96K prompts, ~4.6 chars per token
        return "\n\n".join(long[1:4])
    raise SystemExit(f"unknown context {context}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--contexts", nargs="+", default=["short", "96k", "290k"])
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    harness.MODEL = "glm53-w4a16-tiered"
    report = {}
    for context in args.contexts:
        label = f"decode-{context}"
        print(f"--- {label} {time.strftime('%T')} ---", flush=True)
        prompt_tokens = harness.metrics().get("prompt_tokens_total", 0.0)
        thread = threading.Thread(
            target=mimo.completion,
            args=(prompt_for(context), 3000, args.temperature), daemon=True)
        thread.start()
        harness.wait_for_decode(1, timeout=1500)
        prefill = harness.metrics().get("prompt_tokens_total", 0.0) - prompt_tokens
        time.sleep(1.0)
        before = mimo.counters()
        time.sleep(6.0)
        unprofiled = mimo.rate(before, mimo.counters())
        harness.post("/start_profile")
        before = mimo.counters()
        time.sleep(args.window)
        profiled = mimo.rate(before, mimo.counters())
        try:
            harness.post("/stop_profile")
        except Exception as error:  # workers still flush their traces
            print(f"stop_profile failed, continuing: {error}", flush=True)
        thread.join()
        time.sleep(45)
        files = harness.move_traces(args.trace_root, label)
        report[label] = {"prompt_tokens": prefill, "unprofiled": unprofiled,
                         "profiled_window": profiled, "trace_files": files}
        print(json.dumps(report[label]), flush=True)
        (args.trace_root / "capture-report.json").write_text(
            json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
