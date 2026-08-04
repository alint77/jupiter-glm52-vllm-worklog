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
MODEL = "glm52-w4a16-tiered"


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


def wait_for_decode(num_running: int, timeout: float = 180.0) -> None:
    """Block until `num_running` requests are past prefill and decoding."""
    deadline = time.time() + timeout
    last_prompt = None
    stable = 0
    while time.time() < deadline:
        values = metrics()
        running = values.get("num_requests_running", 0)
        prompt_total = values.get("prompt_tokens_total", 0)
        if running >= num_running and prompt_total == last_prompt:
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        last_prompt = prompt_total
        time.sleep(0.25)
    raise SystemExit(f"never reached {num_running} decoding requests")


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

    # 1. Prefill, c1: profiler opens before the request, so the window holds the
    #    two 8,192-token chunks of one 16K prompt and nothing else.
    run_capture("prefill-c1", fire([0], 1), settle=0.0, window=12.0)

    # 2. Decode, c1: wait until the prompt is fully ingested, then capture.
    def decode_c1():
        threads = fire([1], 512)()
        wait_for_decode(1)
        return threads

    run_capture("decode-c1", decode_c1, settle=0.0, window=0.8)

    # 3. Decode, c4: all four past prefill before the window opens.
    def decode_c4():
        threads = fire([2, 3, 4, 5], 512)()
        wait_for_decode(4)
        return threads

    run_capture("decode-c4", decode_c4, settle=0.0, window=1.2)

    # 4. Mixed, c4: eight requests against max_num_seqs=4, so the window that
    #    opens once the first four are decoding also contains the admission and
    #    chunked prefill of the next ones displacing decode.
    def mixed_c4():
        threads = fire([6, 7, 8, 9, 10, 11, 12, 13], 512)()
        wait_for_decode(4)
        return threads

    run_capture("mixed-c4", mixed_c4, settle=6.0, window=5.0)

    (trace_root / "capture-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
