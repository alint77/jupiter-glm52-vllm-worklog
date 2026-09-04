"""Drive bounded torch-profiler captures on a running tiered server.

Each capture waits for a known engine state before opening the profiler window,
so the trace contains the phase it is named for rather than whatever the clock
happened to land on.
"""

import argparse
import json
import shutil
import threading
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8027"
MODEL = "glm53-cmp-tiered"


def post(path: str, payload: dict | None = None, timeout: float = 600.0):
    data = json.dumps(payload).encode() if payload is not None else b""
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body[:1] in (b"{", b"[") else None


def metrics() -> dict[str, float]:
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=30) as response:
        text = response.read().decode()
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        key, _, value = line.rpartition(" ")
        try:
            values[key.split("{")[0].replace("vllm:", "")] = float(value)
        except ValueError:
            pass
    return values


def completion(prompt: str, max_tokens: int) -> None:
    post(
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
    )


def wait_for_decode(num_running: int, timeout: float = 600.0) -> None:
    """Block until `num_running` requests are past prefill and decoding.

    Gated on generation_tokens_total, which advances once per decode step.
    The previous version watched prompt_tokens_total, which only advances
    when a prefill *chunk completes*; a chunk takes seconds, so two polls a
    quarter second apart read as "stable" in the middle of one and the
    window opened inside prefill. That accident is documented in
    2026-08-05-prod-profile and is the bug this avoids.
    """
    deadline = time.time() + timeout
    last_gen = None
    rising = 0
    while time.time() < deadline:
        values = metrics()
        running = values.get("num_requests_running", 0)
        gen = values.get("generation_tokens_total", 0)
        if running >= num_running and last_gen is not None and gen > last_gen:
            rising += 1
            if rising >= 2:
                return
        else:
            rising = 0
        last_gen = gen
        time.sleep(0.25)
    raise SystemExit(f"never reached {num_running} decoding requests")


def wait_for_running(num_running: int, timeout: float = 300.0) -> None:
    """Block until `num_running` requests are scheduled on the engine.

    Prefill is measured by opening the window on the request rather than on
    the clock: the POST carries a 441K-character prompt, and tokenizing and
    admitting it is seconds of host work that would otherwise be charged to
    the window and displace the chunks it is meant to hold.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if metrics().get("num_requests_running", 0) >= num_running:
            return
        time.sleep(0.1)
    raise SystemExit(f"never reached {num_running} running requests")


def move_traces(trace_root: Path, label: str) -> int:
    target = trace_root / label
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path in sorted(trace_root.glob("*.pt.trace.json*")):
        shutil.move(str(path), target / path.name)
        moved += 1
    return moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--trace-root", required=True)
    args = parser.parse_args()

    prompts = [
        json.loads(line)["prompt"]
        for line in Path(args.prompts).read_text().splitlines()
    ]
    trace_root = Path(args.trace_root)
    report: dict[str, dict] = {}

    def run_capture(label: str, body, settle: float, window: float) -> None:
        print(f"--- {label} ---", flush=True)
        before = metrics()
        threads = body()
        if settle:
            time.sleep(settle)
        start = time.time()
        post("/start_profile")
        time.sleep(window)
        try:
            post("/stop_profile")
        except Exception as error:  # the workers still flush their traces
            print(f"stop_profile failed, continuing: {error}", flush=True)
        elapsed = time.time() - start
        for thread in threads:
            thread.join()
        # A large trace takes tens of seconds to serialize, and opening the next
        # window before that finishes fails the profiler RPC.
        time.sleep(45)
        after = metrics()
        files = move_traces(trace_root, label)
        steps = None
        generated = after.get("generation_tokens_total", 0) - before.get(
            "generation_tokens_total", 0
        )
        accepted = after.get("spec_decode_num_accepted_tokens_total", 0) - before.get(
            "spec_decode_num_accepted_tokens_total", 0
        )
        if generated:
            steps = generated - accepted
        report[label] = {
            "window_s": round(elapsed, 3),
            "trace_files": files,
            "generated_tokens": generated,
            "target_steps": steps,
        }
        print(json.dumps(report[label]), flush=True)

    def fire(prompt_indices, max_tokens):
        def body():
            threads = [
                threading.Thread(
                    target=completion, args=(prompts[index], max_tokens), daemon=True
                )
                for index in prompt_indices
            ]
            for thread in threads:
                thread.start()
            return threads

        return body

    # Both captures run the same ~96K Claude-Code-shaped prompt, so decode is
    # profiled at the context length it actually serves. Profiling decode off a
    # short prompt would understate attention, which grows with context.

    # 1. Prefill. The profiler opens before the request, so the window holds
    #    chunked prefill and nothing else. Chunks are 8192 tokens and
    #    structurally identical, so a window covering a few of them is
    #    representative and keeps the trace to a size that loads; a full 96K
    #    prefill is ~12 chunks and four ranks of that is unwieldy.
    def prefill_c1():
        threads = fire([0], 1)()
        wait_for_running(1)
        return threads

    # ~1.93 s/chunk measured unprofiled (96K in 23.17 s TTFT), so 10 s is
    # about five structurally identical chunks -- representative, and four
    # ranks of it stays near the 14 MB the last prefill capture produced.
    run_capture("prefill", prefill_c1, settle=0.0, window=10.0)

    # 2. Decode at ~96K context. max_tokens is large enough that the request is
    #    still decoding after the window, and wait_for_decode gates on
    #    generation tokens so the window cannot open inside prefill.
    def decode_c1():
        threads = fire([1], 4096)()
        wait_for_decode(1)
        return threads

    # ~29 ms/step at MTP3 c=1, so 2 s is ~70 steps -- plenty of steady state
    # without a trace that takes minutes to serialise.
    run_capture("decode", decode_c1, settle=0.0, window=2.0)

    (trace_root / "capture-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
