"""Exercise the Anthropic Messages API the way Claude Code does.

Checks, against a running server: a tool call comes back as a tool_use block
with parsed JSON input, a tool_result round-trips to a text answer, the
streamed form carries the same tool_use, and reasoning arrives as a thinking
block rather than leaking into text.

Usage: tool_probe.py <base_url> <token_file> <model>
"""

import json
import sys
import urllib.request

base_url, token_file, model = sys.argv[1:4]
token = open(token_file).read().strip()

TOOLS = [
    {
        "name": "Bash",
        "description": "Run a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command"},
                "description": {"type": "string"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file from the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
]
SYSTEM = "You are a coding agent. Use the tools to act; do not guess outputs."
ASK = "How many lines does /etc/hosts have? Use a tool to find out."


def post(body: dict, stream: bool = False):
    body = {"model": model, "max_tokens": 2048, **body, "stream": stream}
    request = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": token,
            "authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
        },
    )
    response = urllib.request.urlopen(request, timeout=600)
    if not stream:
        return json.load(response)
    return [line.decode().rstrip() for line in response if line.strip()]


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


results = []
first = post(
    {
        "system": SYSTEM,
        "tools": TOOLS,
        "messages": [{"role": "user", "content": ASK}],
    }
)
kinds = [block["type"] for block in first["content"]]
print("blocks:", kinds, "stop_reason:", first.get("stop_reason"))
tool_uses = [b for b in first["content"] if b["type"] == "tool_use"]
text = " ".join(b.get("text", "") for b in first["content"] if b["type"] == "text")
results.append(check("tool_use block", bool(tool_uses), json.dumps(tool_uses)[:300]))
results.append(check("stop_reason tool_use", first.get("stop_reason") == "tool_use"))
results.append(
    check("no raw tool markup in text", "<tool_call>" not in text and "<function=" not in text)
)
results.append(check("no <think> in text", "<think>" not in text and "</think>" not in text))
results.append(check("thinking block present", "thinking" in kinds, "(optional)"))

if tool_uses:
    call = tool_uses[0]
    results.append(
        check(
            "tool input parsed",
            isinstance(call["input"], dict) and bool(call["input"]),
            json.dumps(call["input"]),
        )
    )
    second = post(
        {
            "system": SYSTEM,
            "tools": TOOLS,
            "messages": [
                {"role": "user", "content": ASK},
                {"role": "assistant", "content": first["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": "12 /etc/hosts",
                        }
                    ],
                },
            ],
        }
    )
    answer = " ".join(b.get("text", "") for b in second["content"] if b["type"] == "text")
    print("follow-up:", answer[:300])
    results.append(check("tool_result round trip mentions 12", "12" in answer))

events = post(
    {"system": SYSTEM, "tools": TOOLS, "messages": [{"role": "user", "content": ASK}]},
    stream=True,
)
payloads = [json.loads(e[6:]) for e in events if e.startswith("data: ")]
starts = [
    p["content_block"]["type"] for p in payloads if p.get("type") == "content_block_start"
]
partial = "".join(
    p["delta"].get("partial_json", "")
    for p in payloads
    if p.get("type") == "content_block_delta"
    and p["delta"].get("type") == "input_json_delta"
)
print("stream blocks:", starts, "tool input:", partial[:200])
results.append(check("stream tool_use block", "tool_use" in starts))
try:
    results.append(check("stream tool input is JSON", bool(json.loads(partial or "null"))))
except json.JSONDecodeError:
    results.append(check("stream tool input is JSON", False, partial[:200]))

required = [r for i, r in enumerate(results) if i != 4]
print("ALL REQUIRED PASS" if all(required) else "SOME CHECKS FAILED")
