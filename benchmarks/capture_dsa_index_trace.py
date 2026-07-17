#!/usr/bin/env python3

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from vllm.model_executor.model_loader.tiered_moe_manifest import (
    build_glm_w4a16_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8022")
    parser.add_argument("--model", default="glm52-w4a16-tiered")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=32768)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=154880)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def post_completion(url: str, body: dict, request_id: str) -> dict:
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def load_samples(trace_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(trace_dir.glob("sample-*.npz")):
        with np.load(path) as data:
            indices = data["indices"].astype(np.int32)
            request_id = str(data["request_id"])
            position = int(data["context_position"])
        if indices.shape != (21, 2048):
            raise ValueError(f"Unexpected DSA sample shape in {path}: {indices.shape}")
        if indices.min() < 0 or indices.max() > position:
            raise ValueError(f"Out-of-range DSA index in {path}")
        if (np.diff(np.sort(indices, axis=1), axis=1) == 0).any():
            raise ValueError(f"Duplicate position in DSA top-k set in {path}")
        samples.append(
            {
                "file": path.name,
                "request_id": request_id,
                "context_position": position,
                "valid_indices": int((indices >= 0).sum()),
            }
        )
    return samples


def main() -> None:
    args = parse_args()
    if not args.trace_dir.joinpath("header.json").is_file():
        raise ValueError("Trace server has not initialized the output directory")
    if list(args.trace_dir.glob("sample-*.npz")):
        raise ValueError("Trace directory already contains samples")

    requests = []
    for index in range(args.num_requests):
        rng = np.random.default_rng(args.seed + index)
        prompt = rng.integers(
            1000,
            args.vocab_size - 1000,
            size=args.prompt_len,
            dtype=np.int32,
        )
        request_hash = hashlib.sha256(prompt.tobytes()).hexdigest()
        request_id = f"dsa-{request_hash[:20]}"
        started = time.monotonic()
        result = post_completion(
            args.base_url,
            {
                "model": args.model,
                "prompt": prompt.tolist(),
                "max_tokens": args.output_len,
                "temperature": 0,
                "ignore_eos": True,
                "cache_salt": request_hash,
            },
            request_id,
        )
        usage = result["usage"]
        requests.append(
            {
                "request_hash": request_hash,
                "request_id": request_id,
                "prompt_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "elapsed_seconds": time.monotonic() - started,
                "domain": "synthetic-random-token",
            }
        )
        print(f"captured request {index + 1}/{args.num_requests}")

    samples = load_samples(args.trace_dir)
    for request in requests:
        matched = [
            sample
            for sample in samples
            if request["request_id"] in sample["request_id"]
        ]
        if not matched:
            raise ValueError(f"No DSA samples for {request['request_id']}")
        request["samples"] = matched

    model_manifest = build_glm_w4a16_manifest(args.model_path)
    header_path = args.trace_dir / "header.json"
    header = json.loads(header_path.read_text())
    header.update(
        {
            "config_sha256": model_manifest.config_sha256,
            "index_sha256": model_manifest.index_sha256,
            "request_count": len(requests),
            "sample_count": len(samples),
        }
    )
    header_path.write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")
    (args.trace_dir / "manifest.json").write_text(
        json.dumps(requests, indent=2, sort_keys=True) + "\n"
    )
    print(f"captured {len(samples)} sampled context rows")


if __name__ == "__main__":
    main()
