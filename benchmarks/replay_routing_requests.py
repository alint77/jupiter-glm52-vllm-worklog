#!/usr/bin/env python3

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="glm52-w4a16-tiered")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=154880)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--cache-salt-suffix", default="")
    return parser.parse_args()


def make_prompt(seed: int, prompt_len: int, vocab_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1000, vocab_size - 1000, size=prompt_len, dtype=np.int32)


def stream_request(base_url: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    content_times = []
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                content_times.append(time.perf_counter())
    finished = time.perf_counter()
    if not content_times:
        raise RuntimeError("Streaming replay returned no content chunks")
    output_tokens = body["max_tokens"]
    if output_tokens < 2:
        raise ValueError("Latency replay requires at least two output tokens")
    return {
        "ttft_ms": (content_times[0] - started) * 1000,
        "tpot_ms": (finished - content_times[0]) * 1000 / (output_tokens - 1),
        "total_ms": (finished - started) * 1000,
        "content_chunks": len(content_times),
    }


def main() -> None:
    args = parse_args()
    common = {
        "model": args.model,
        "max_tokens": args.output_len,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    for index in range(args.warmup_requests):
        prompt = make_prompt(
            args.seed + args.num_requests + index,
            args.prompt_len,
            args.vocab_size,
        )
        stream_request(
            args.base_url,
            common
            | {
                "prompt": prompt.tolist(),
                "cache_salt": f"warmup-{args.seed}-{index}",
            },
        )

    results = []
    for index in range(args.num_requests):
        prompt = make_prompt(args.seed + index, args.prompt_len, args.vocab_size)
        request_hash = hashlib.sha256(prompt.tobytes()).hexdigest()
        result = stream_request(
            args.base_url,
            common
            | {
                "prompt": prompt.tolist(),
                "cache_salt": request_hash + args.cache_salt_suffix,
            },
        )
        result.update(
            {
                "request_index": index,
                "request_hash": request_hash,
                "prompt_tokens": args.prompt_len,
                "output_tokens": args.output_len,
            }
        )
        results.append(result)
        print(
            f"replayed {index + 1}/{args.num_requests}: "
            f"TTFT {result['ttft_ms']:.2f} ms, TPOT {result['tpot_ms']:.3f} ms"
        )
    args.output.write_text(json.dumps({"requests": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
