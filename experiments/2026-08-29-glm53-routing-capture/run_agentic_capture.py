#!/usr/bin/env python3
"""Drive agentic coding turns against the GLM-5.3 routing-capture host.

Each assistant turn is one request, and the server writes one routed-expert
trace per request, so the workload the model *generates* -- tool-call JSON,
code, and reasoning about code -- is what ends up in the ranking. Reads are
served from the real vLLM tree so tool results carry genuine source; writes are
confined to a sandbox directory.

The loop is resumable: completed task ids are appended to ``done.txt`` in the
output directory and skipped on a re-run.
"""

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

MAX_TOOL_ROUNDS = 8
MAX_RESULT_CHARS = 6000

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the entries of a directory in the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file, optionally a line range (1-indexed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search the repository for a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file into the scratch sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a coding agent working inside the vLLM repository at \
/e/project1/profound/alint77/vllm. You have tools to list directories, read \
files, search with regular expressions, and write files into a scratch sandbox.

Work the way an engineer does: search before you read, read before you answer, \
and ground every claim in code you have actually opened. Cite files as \
path:line. When you write code, match the surrounding style, keep lines under \
88 characters, and use Google-style docstrings. Prefer several targeted tool \
calls over one broad one. When you have enough to answer, stop calling tools \
and give the answer."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="glm53-nvfp4-tiered")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/e/project1/profound/alint77/vllm"))
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args()


def _safe_repo_path(repo: Path, raw: str) -> Path:
    candidate = (repo / raw.lstrip("/")).resolve()
    if not str(candidate).startswith(str(repo.resolve())):
        raise ValueError(f"path escapes the repository: {raw}")
    return candidate


def run_tool(name: str, arguments: dict, repo: Path, sandbox: Path) -> str:
    try:
        if name == "list_dir":
            target = _safe_repo_path(repo, arguments.get("path", "."))
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            return "\n".join(entries[:200]) or "(empty)"
        if name == "read_file":
            target = _safe_repo_path(repo, arguments["path"])
            lines = target.read_text(errors="replace").splitlines()
            start = max(1, int(arguments.get("start_line") or 1))
            end = min(len(lines), int(arguments.get("end_line") or start + 199))
            return "\n".join(
                f"{n}\t{lines[n - 1]}" for n in range(start, end + 1)
            ) or "(no lines in range)"
        if name == "grep":
            scope = _safe_repo_path(repo, arguments.get("path", "."))
            command = ["grep", "-rnE", arguments["pattern"], str(scope)]
            if arguments.get("glob"):
                command.insert(2, f"--include={arguments['glob']}")
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=60
            )
            out = result.stdout or "(no matches)"
            return "\n".join(out.splitlines()[:120])
        if name == "write_file":
            target = (sandbox / arguments["path"].lstrip("/")).resolve()
            if not str(target).startswith(str(sandbox.resolve())):
                raise ValueError("path escapes the sandbox")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(arguments["content"])
            return f"wrote {len(arguments['content'])} bytes to {target}"
    except Exception as error:  # surfaced to the model as a tool result
        return f"ERROR: {type(error).__name__}: {error}"
    return f"ERROR: unknown tool {name}"


def post_chat(args: argparse.Namespace, messages: list[dict]) -> dict:
    body = json.dumps(
        {
            "model": args.model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    ).encode()
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read())


def run_turn(
    args: argparse.Namespace, messages: list[dict], log, record
) -> int:
    """Run one user turn to completion, returning the number of requests made."""
    requests_made = 0
    for _ in range(MAX_TOOL_ROUNDS):
        payload = post_chat(args, messages)
        requests_made += 1
        record(payload.get("id", ""))
        choice = payload["choices"][0]
        message = choice["message"]
        usage = payload.get("usage", {})
        calls = message.get("tool_calls") or []
        log(
            f"    request {requests_made}: "
            f"{usage.get('completion_tokens', 0)} out tokens, "
            f"{len(calls)} tool call(s)"
        )
        messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                **({"tool_calls": calls} if calls else {}),
            }
        )
        if not calls:
            return requests_made
        for call in calls:
            function = call["function"]
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as error:
                result = f"ERROR: arguments were not valid JSON: {error}"
            else:
                result = run_tool(function["name"], arguments, args.repo, args.sandbox)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result[:MAX_RESULT_CHARS],
                }
            )
    return requests_made


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.sandbox.mkdir(parents=True, exist_ok=True)
    done_path = args.output_dir / "done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    tasks = json.loads(args.tasks.read_text())

    log_path = args.output_dir / "driver.log"

    def log(line: str) -> None:
        stamped = f"[{time.strftime('%H:%M:%S')}] {line}"
        print(stamped, flush=True)
        with log_path.open("a") as file:
            file.write(stamped + "\n")

    attribution_path = args.output_dir / "requests.jsonl"

    def make_recorder(task_id: str):
        def record(request_id: str) -> None:
            if not request_id:
                return
            with attribution_path.open("a") as file:
                file.write(
                    json.dumps({"request_id": request_id, "domain": task_id})
                    + "\n"
                )

        return record

    total_requests = 0
    for pass_index in range(args.passes):
        for task in tasks:
            key = f"{task['id']}#{pass_index}"
            if key in done:
                continue
            turns = task.get("parts") or [task["prompt"]]
            log(f"task {key}: {len(turns)} turn(s)")
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            try:
                for turn_index, turn in enumerate(turns):
                    messages.append({"role": "user", "content": turn})
                    log(f"  turn {turn_index + 1}/{len(turns)}")
                    total_requests += run_turn(
                        args, messages, log, make_recorder(task["id"])
                    )
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                log(f"  task {key} aborted: {type(error).__name__}: {error}")
                continue
            with done_path.open("a") as file:
                file.write(key + "\n")
            log(f"task {key} done; {total_requests} requests so far")
    log(f"finished: {total_requests} requests")


if __name__ == "__main__":
    main()
