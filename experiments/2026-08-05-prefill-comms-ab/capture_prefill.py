"""Capture one profiled prefill chunk, then move the traces under a label."""

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8027"
MODEL = "glm52-w4a16-tiered"


def post(path, payload=None, timeout=900.0):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body[:1] in (b"{", b"[") else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--trace-root", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--prompt-index", type=int, default=15)
    args = p.parse_args()

    prompt = json.loads(
        Path(args.prompts).read_text().splitlines()[args.prompt_index]
    )["prompt"]
    root = Path(args.trace_root)

    # Profiler opens before the request, so the window holds only prefill:
    # max_tokens=1 means the single decode step is negligible after it.
    post("/start_profile")
    start = time.time()
    post("/v1/completions", {
        "model": MODEL, "prompt": prompt, "max_tokens": 1,
        "temperature": 0, "ignore_eos": True,
    })
    elapsed = time.time() - start
    try:
        post("/stop_profile")
    except Exception as exc:  # workers still flush
        print(f"stop_profile failed, continuing: {exc}", flush=True)
    time.sleep(45)

    target = root / args.label
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path in sorted(root.glob("*.pt.trace.json*")):
        shutil.move(str(path), target / path.name)
        moved += 1
    print(json.dumps({"label": args.label, "prefill_s": round(elapsed, 3),
                      "trace_files": moved}))


if __name__ == "__main__":
    main()
