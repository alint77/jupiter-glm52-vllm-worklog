#!/usr/bin/env python3
"""Real Claude Code requests from the proxy's upstream capture -> a replay set.

Keeps the Anthropic bodies exactly as Claude Code sent them ("original" phase),
main-agent calls only (max_tokens >= 8192; the 64-token side calls are titles
and the like), in time order, tagged with their session.

    extract_cc.py --out /e/fscratch/.../cc-replay.jsonl
"""

import argparse
import glob
import json
from pathlib import Path

CAPTURE = "/e/project1/profound/alint77/vllm/proxy/.run/capture"


def session(body: dict) -> str:
    uid = (body.get("metadata") or {}).get("user_id", "")
    try:
        return json.loads(uid).get("session_id", uid)
    except (json.JSONDecodeError, AttributeError):
        return uid.split("_session_")[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(f"{CAPTURE}/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["phase"] != "original":
                continue
            body = r["data"] if isinstance(r["data"], dict) else json.loads(r["data"])
            if (body.get("max_tokens") or 0) >= 8192:
                rows.append({"ts": r["timestamp"], "session": session(body), "body": body})
    rows.sort(key=lambda r: r["ts"])
    with args.out.open("w") as out:
        for i, r in enumerate(rows):
            out.write(json.dumps({"index": i, **r}) + "\n")
    print(f"{len(rows)} requests -> {args.out}")


if __name__ == "__main__":
    main()
