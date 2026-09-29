#!/usr/bin/env python3
"""Decode step time on real agentic coding traffic, the same for every model.

Replays the capture task set (../2026-09-26-mimo-routing-profile/tasks-*.json)
with the capture driver's system prompt, tools and agent loop
(../2026-08-29-glm53-routing-capture/run_agentic_capture.py), through
/v1/chat/completions at the models' shared defaults (temperature 1.0, top_p
0.95) and a seed per request. Around each request it diffs the server's
Prometheus counters, so every row is server-side and per request at c=1:

    decode_s     vllm:request_decode_time_seconds (first token -> finish)
    steps        vllm:spec_decode_num_drafts (one per verify step)
    accepted     vllm:spec_decode_num_accepted_tokens

    agentic_bench.py --base-url http://127.0.0.1:8027 --model glm53 \\
        --tasks tasks-0.json --out rows.jsonl [--passes 1] [--limit-requests N]
"""

import argparse
import hashlib
import importlib.util
import json
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "agentic", HERE.parent / "2026-08-29-glm53-routing-capture/run_agentic_capture.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

METRICS = {
    "decode_s": "vllm:request_decode_time_seconds_sum",
    "prefill_s": "vllm:request_prefill_time_seconds_sum",
    "requests": "vllm:request_decode_time_seconds_count",
    "steps": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}


def scrape(base: str) -> dict[str, float]:
    text = urllib.request.urlopen(f"{base}/metrics", timeout=30).read().decode()
    out = dict.fromkeys(METRICS, 0.0)
    for line in text.splitlines():
        m = re.match(r"^([a-z_:]+)(?:\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if not m:
            continue
        for key, name in METRICS.items():
            if m.group(1) == name:
                out[key] += float(m.group(2))
    return out


def post(args, messages: list[dict], seed: int) -> dict:
    body = {"model": args.model, "messages": messages, "tools": A.TOOLS,
            "tool_choice": "auto", "max_tokens": args.max_tokens,
            "temperature": 1.0, "top_p": 0.95, "seed": seed}
    request = urllib.request.Request(
        f"{args.base_url}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {args.api_key}"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read())


class Profiler:
    """Up to `count` torch-profiler windows, each opened ~1 s into a request's
    decode and held `window` s (or until it ends); each window's traces move to
    trace_root/window-<i> so they can be analysed one window at a time."""

    def __init__(self, base: str, trace_root: Path | None, count: int, window: float,
                 min_steps: int = 40):
        self.base, self.root, self.left, self.window = base, trace_root, count, window
        self.min_steps = min_steps
        self.done, self.last = 0, False

    def call(self, path: str) -> None:
        urllib.request.urlopen(urllib.request.Request(f"{self.base}{path}", method="POST"),
                               timeout=600).read()

    def around(self, fn):
        self.last = False
        if not self.left or self.root is None:
            return fn()
        result, error = {}, {}

        def run():
            try:
                result["v"] = fn()
            except Exception as e:  # re-raised below
                error["e"] = e

        thread = threading.Thread(target=run)
        base = scrape(self.base)["steps"]
        thread.start()
        deadline = time.monotonic() + 120
        while thread.is_alive() and scrape(self.base)["steps"] < base + 30:
            if time.monotonic() > deadline:
                break
            time.sleep(0.05)
        if thread.is_alive():
            before = set(self.root.rglob("*.gz"))
            s0 = scrape(self.base)["steps"]
            self.call("/start_profile")
            end = time.monotonic() + self.window
            while thread.is_alive() and time.monotonic() < end:
                time.sleep(0.05)
            steps = scrape(self.base)["steps"] - s0
            self.call("/stop_profile")
            time.sleep(5)  # let every rank finish writing
            new = set(self.root.glob("*.gz")) - before
            if steps < self.min_steps:  # the request ended too soon: not a decode window
                for f in new:
                    f.unlink()
                print(f"discarded a window with {steps:.0f} steps", flush=True)
                thread.join()
                if error:
                    raise error["e"]
                return result["v"]
            dest = self.root / f"window-{self.done}"
            dest.mkdir(parents=True, exist_ok=True)
            for f in new:
                f.rename(dest / f.name)
            self.done += 1
            self.left -= 1
            self.last = True
            print(f"profiled window {self.done} ({steps:.0f} steps) -> {dest}", flush=True)
        thread.join()
        if error:
            raise error["e"]
        return result["v"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8027")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("/e/project1/profound/alint77/vllm"))
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-requests-per-turn", type=int, default=A.MAX_TOOL_ROUNDS)
    ap.add_argument("--limit-requests", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--profile", type=int, default=0, help="profiler windows to take")
    ap.add_argument("--profile-window", type=float, default=2.0)
    ap.add_argument("--profile-min-steps", type=int, default=40)
    ap.add_argument("--trace-root", type=Path, default=None)
    args = ap.parse_args()
    profiler = Profiler(args.base_url, args.trace_root, args.profile, args.profile_window,
                        args.profile_min_steps)
    tasks = [t for f in args.tasks for t in json.loads(f.read_text())]
    sandbox = Path(tempfile.mkdtemp(prefix="agentic-bench-"))
    n = 0
    with args.out.open("a") as out:
        for p in range(args.passes):
            for task in tasks:
                messages = [{"role": "system", "content": A.SYSTEM_PROMPT}]
                for ti, turn in enumerate(task.get("parts") or [task["prompt"]]):
                    messages.append({"role": "user", "content": turn})
                    for ri in range(args.max_requests_per_turn):
                        key = f"{task['id']}#{p}/{ti}/{ri}"
                        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                        before, t0 = scrape(args.base_url), time.monotonic()
                        try:
                            reply = profiler.around(lambda: post(args, messages, seed))
                        except urllib.error.HTTPError as error:
                            out.write(json.dumps({"key": key, "error": f"{error.code}: "
                                                  f"{error.read()[:300]!r}"}) + "\n")
                            out.flush()
                            print(f"{key}: HTTP {error.code}", flush=True)
                            break
                        wall, after = time.monotonic() - t0, scrape(args.base_url)
                        d = {k: after[k] - before[k] for k in METRICS}
                        usage = reply.get("usage", {})
                        msg = reply["choices"][0]["message"]
                        calls = msg.get("tool_calls") or []
                        row = {"key": key, "seed": seed, "wall_s": round(wall, 3),
                               "prompt_tokens": usage.get("prompt_tokens"),
                               "completion_tokens": usage.get("completion_tokens"),
                               "tool_calls": len(calls), "profiled": profiler.last,
                               **{k: round(v, 4) for k, v in d.items()}}
                        if d["steps"] > 0:
                            row["step_ms"] = round(1000 * d["decode_s"] / d["steps"], 3)
                            row["tokens_per_step"] = round(1 + d["accepted"] / d["steps"], 3)
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        n += 1
                        print(f"{key}: ctx {row['prompt_tokens']} out "
                              f"{row['completion_tokens']} step {row.get('step_ms')} ms "
                              f"x {row.get('tokens_per_step')} tok", flush=True)
                        if args.limit_requests and n >= args.limit_requests:
                            return
                        messages.append({"role": "assistant",
                                         "content": msg.get("content") or "",
                                         **({"tool_calls": calls} if calls else {})})
                        if not calls:
                            break
                        for call in calls:
                            fn = call["function"]
                            try:
                                arguments = json.loads(fn.get("arguments") or "{}")
                            except json.JSONDecodeError:
                                arguments = {}
                            result = A.run_tool(fn["name"], arguments, args.repo, sandbox)
                            messages.append({"role": "tool",
                                             "tool_call_id": call.get("id", ""),
                                             "content": result[:A.MAX_RESULT_CHARS]})


if __name__ == "__main__":
    main()
