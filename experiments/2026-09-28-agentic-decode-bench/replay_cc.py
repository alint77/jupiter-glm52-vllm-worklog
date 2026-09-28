#!/usr/bin/env python3
"""Decode step time on real Claude Code traffic, identical for every model.

Replays extract_cc.py's requests, in time order, to the server's Anthropic
endpoint (/v1/messages) exactly as Claude Code sent them, overriding only the
model name, non-streaming, and a cap on output length. Sampling is left to
the server's defaults (both GLM-5.3 and MiMo-V2.6: temperature 1.0, top_p
0.95). Around each request it diffs the server's counters, so each row is
server-side, per request, at c=1:

    decode_s    vllm:request_decode_time_seconds (first token -> finish)
    steps       vllm:spec_decode_num_drafts (one per verify step)
    accepted    vllm:spec_decode_num_accepted_tokens

    replay_cc.py --model glm53 --out rows.jsonl [--stride 5] [--max-tokens 2048]
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

REPLAY = Path("/e/fscratch/profound/naeimitabiei1/agentic-bench/cc-replay.jsonl")
METRICS = {
    "decode_s": "vllm:request_decode_time_seconds_sum",
    "prefill_s": "vllm:request_prefill_time_seconds_sum",
    "steps": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}


def scrape(base: str) -> dict[str, float]:
    text = urllib.request.urlopen(f"{base}/metrics", timeout=30).read().decode()
    out = dict.fromkeys(METRICS, 0.0)
    for line in text.splitlines():
        m = re.match(r"^([a-z_:]+)(?:\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if m:
            for key, name in METRICS.items():
                if m.group(1) == name:
                    out[key] += float(m.group(2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8027")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replay", type=Path, default=REPLAY)
    ap.add_argument("--stride", type=int, default=5, help="every n-th request")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()
    done = 0
    with args.replay.open() as src, args.out.open("a") as out:
        for line in src:
            r = json.loads(line)
            if (r["index"] - args.offset) % args.stride:
                continue
            body = dict(r["body"], model=args.model, stream=False,
                        max_tokens=min(r["body"]["max_tokens"], args.max_tokens))
            request = urllib.request.Request(
                f"{args.base_url}/v1/messages", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "x-api-key": "none",
                         "anthropic-version": "2023-06-01"})
            before, t0 = scrape(args.base_url), time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=args.timeout) as resp:
                    reply = json.loads(resp.read())
            except urllib.error.HTTPError as error:
                row = {"index": r["index"], "session": r["session"][-12:],
                       "error": f"{error.code}: {error.read()[:300]!r}"}
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(row, flush=True)
                continue
            wall, after = time.monotonic() - t0, scrape(args.base_url)
            d = {k: after[k] - before[k] for k in METRICS}
            usage = reply.get("usage", {})
            blocks = [b.get("type") for b in reply.get("content", [])]
            row = {"index": r["index"], "session": r["session"][-12:],
                   "wall_s": round(wall, 3),
                   "context": usage.get("input_tokens"),
                   "output": usage.get("output_tokens"),
                   "stop": reply.get("stop_reason"), "blocks": blocks,
                   **{k: round(v, 4) for k, v in d.items()}}
            if d["steps"] > 0:
                row["step_ms"] = round(1000 * d["decode_s"] / d["steps"], 3)
                row["tokens_per_step"] = round(1 + d["accepted"] / d["steps"], 3)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"#{r['index']} {row['session']}: ctx {row['context']} out "
                  f"{row['output']} ({row['stop']}) step {row.get('step_ms')} ms x "
                  f"{row.get('tokens_per_step')} tok, prefill {d['prefill_s']:.1f} s",
                  flush=True)
            done += 1
            if args.limit and done >= args.limit:
                break


if __name__ == "__main__":
    main()
