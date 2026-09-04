#!/usr/bin/env python3
"""Stream the comparison corpus at one arm and record per-request metrics.

Sequential requests, matching the c=1 server shape. Per request: TTFT from
the first streamed chunk, decode time from the last, and the /metrics deltas
bracketing it, so acceptance-by-task-kind (code vs prose) falls out of the
same run. Four temperature-0 probes run first; speculative decoding is
lossless, so their outputs must be identical on every arm, which the
aggregator checks.

Sized from the live 2026-09-04 server: ~13.4k tok/s prefill, 60-74 tok/s
decode at 150K context, ~2.5 tokens/step at K=7.
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

PROBES = [
    "What is 17*23? Answer with just the number.",
    "In one sentence: why does a chained per-block hash miss everything "
    "after the first divergent block?",
    "Name the three tiered planner fixed HBM allocations in order.",
    "Reply with exactly: probe ok 4",
]

BASE = "http://127.0.0.1:8027"


def auth_headers() -> dict:
    key = os.environ.get("VLLM_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def get(path: str) -> str:
    req = urllib.request.Request(BASE + path, headers=auth_headers())
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def spec_counters() -> tuple[float, float, dict]:
    drafts = accepted = 0.0
    per_pos = {}
    for line in get("/metrics").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        base = name.split("{", 1)[0]
        if base.startswith("vllm:spec_decode_num_drafts"):
            drafts += float(value)
        elif base.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            pos = name.split('position="', 1)[1].split('"', 1)[0]
            per_pos[pos] = per_pos.get(pos, 0.0) + float(value)
        elif base.startswith("vllm:spec_decode_num_accepted_tokens"):
            accepted += float(value)
    return drafts, accepted, per_pos


def stream_chat(messages: list[dict], model: str, max_tokens: int,
                temperature: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"} | auth_headers(),
    )
    t_open = time.perf_counter()
    ttft = None
    t_prev = None
    step_gaps = []
    content_parts, reason_parts = [], []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if "usage" in d and d["usage"]:
                usage = d["usage"]
            for choice in d.get("choices", []):
                delta = choice.get("delta") or {}
                # Streaming deltas carry the thinking channel as "reasoning";
                # non-streaming responses call it "reasoning_content".
                if delta.get("reasoning"):
                    delta.setdefault("reasoning_content", delta["reasoning"])
                if delta.get("content") or delta.get("reasoning_content"):
                    t_now = time.perf_counter()
                    if ttft is None:
                        ttft = t_now - t_open
                    elif t_prev is not None:
                        # Under spec decoding one SSE chunk carries the tokens
                        # accepted by one verify step, so a chunk gap is a step.
                        step_gaps.append(t_now - t_prev)
                    t_prev = t_now
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reason_parts.append(delta["reasoning_content"])
            if d.get("error"):
                raise RuntimeError(f"stream error: {d['error']}")
    t_close = time.perf_counter()
    return {
        "ttft": ttft,
        "wall": t_close - t_open,
        "content": "".join(content_parts),
        "reasoning_chars": len("".join(reason_parts)),
        "step_gaps": step_gaps,
        "usage": usage,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", required=True, help="dflash | mtp")
    ap.add_argument("--width", type=int, required=True, help="K")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=str(Path(__file__).parent / "corpus.jsonl"))
    ap.add_argument("--model", default="glm53-cmp-tiered")
    ap.add_argument("--host", default="127.0.0.1",
                    help="the arms serve localhost; a remote host is for sanity runs")
    ap.add_argument("--port", type=int, default=8027)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    global BASE
    BASE = f"http://{args.host}:{args.port}"

    probes_out = []
    for p in PROBES:
        r = stream_chat([{"role": "user", "content": p}], args.model, 640, 0.0)
        probes_out.append({"prompt": p, "content": r["content"]})
        print(f"  probe: {r['content'][:60]!r}", flush=True)

    rows = []
    corpus = [json.loads(l) for l in open(args.corpus)]
    # Warmup is the probes; the corpus itself is measured.
    t_start = time.perf_counter()
    for i, row in enumerate(corpus):
        d0, a0, pp0 = spec_counters()
        r = stream_chat(row["messages"], args.model, args.max_tokens,
                        args.temperature)
        d1, a1, pp1 = spec_counters()
        usage = r["usage"] or {}
        out_tok = usage.get("completion_tokens",
                            len(r["content"]) + r["reasoning_chars"])
        steps = d1 - d0
        accepted = a1 - a0
        per_pos = {k: round(v1 - pp0.get(k, 0.0), 1)
                   for k, v1 in pp1.items()
                   if v1 - pp0.get(k, 0.0) > 0}
        gaps = sorted(r["step_gaps"])
        rec = {
            "id": row["id"],
            "task_kind": row["task_kind"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": out_tok,
            "ttft_s": round(r["ttft"], 3) if r["ttft"] else None,
            "decode_s": round(r["wall"] - (r["ttft"] or 0.0), 3),
            "wall_s": round(r["wall"], 3),
            "steps": int(steps),
            "accepted": int(accepted),
            "al": round((accepted + steps) / steps, 4) if steps else None,
            "step_ms_median": round(gaps[len(gaps) // 2] * 1000, 1)
            if gaps else None,
            "per_pos": per_pos,
            "reasoning_chars": r["reasoning_chars"],
            "content_chars": len(r["content"]),
            # The arms serve without --reasoning-parser (run-server.sh sets
            # none and arm.sh does not add one), so the thinking trace stays
            # inline in content and closes with a literal </think>. That tag
            # is the real "did it stop reasoning and answer" signal; a
            # non-empty content is not, since content is never empty here.
            # Generation is unaffected -- the parser only splits the stream.
            "think_closed": "</think>" in r["content"],
            "answer_sample": r["content"].split("</think>")[-1][:400],
            "finished_thinking": "</think>" in r["content"],
        }
        rows.append(rec)
        if (i + 1) % 6 == 0:
            done = [(x["ttft_s"], x["decode_s"], x["output_tokens"]) for x in rows]
            print(f"  {i+1}/{len(corpus)}  last ttft {done[-1][0]}s "
                  f"decode {done[-1][1]}s out {done[-1][2]} tok", flush=True)

    wall = time.perf_counter() - t_start
    out_tok_total = sum(r["output_tokens"] or 0 for r in rows)
    prompts_total = sum(r["prompt_tokens"] or 0 for r in rows)
    result = {
        "label": args.label,
        "mode": args.mode,
        "width": args.width,
        "temperature": args.temperature,
        "requests": len(rows),
        "wall_s": round(wall, 2),
        "prompt_tokens_total": prompts_total,
        "output_tokens_total": out_tok_total,
        "output_tok_per_s": round(out_tok_total / wall, 2),
        "per_request": rows,
        "probes": probes_out,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"done: {out_tok_total} tok in {wall:.0f}s = "
          f"{result['output_tok_per_s']} tok/s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
