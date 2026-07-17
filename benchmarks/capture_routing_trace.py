#!/usr/bin/env python3

import argparse
import base64
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8022")
    parser.add_argument("--model", default="glm52-w4a16-tiered")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=154880)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def post_completion(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def main() -> None:
    args = parse_args()
    if args.num_requests < 2:
        raise ValueError("At least two requests are required for a held-out split")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = []
    for index in range(args.num_requests):
        rng = np.random.default_rng(args.seed + index)
        prompt = rng.integers(
            1000,
            args.vocab_size - 1000,
            size=args.prompt_len,
            dtype=np.int32,
        )
        request_hash = hashlib.sha256(prompt.tobytes()).hexdigest()
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
        )
        encoded = result["choices"][0].get("routed_experts")
        if encoded is None:
            raise RuntimeError("Server response did not include routed experts")
        routes = np.load(io.BytesIO(base64.b64decode(encoded)))
        if routes.ndim != 3 or routes.shape[1:] != (78, 8):
            raise ValueError(f"Unexpected routed-expert shape: {routes.shape}")
        if routes.min() < 0 or routes.max() >= 256:
            raise ValueError("Trace contains an invalid logical expert ID")
        name = f"request-{index:03d}.npz"
        np.savez_compressed(
            args.output_dir / name,
            routes=routes.astype(np.uint16),
            request_hash=np.array(request_hash),
        )
        manifest.append(
            {
                "file": name,
                "request_hash": request_hash,
                "prompt_tokens": args.prompt_len,
                "output_tokens": args.output_len,
                "routed_tokens": int(routes.shape[0]),
                "domain": "synthetic-random-token",
            }
        )
        print(f"captured {index + 1}/{args.num_requests}: {routes.shape}")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
