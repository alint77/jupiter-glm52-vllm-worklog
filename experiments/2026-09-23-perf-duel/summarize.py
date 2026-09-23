"""Summarize an opencode --format json transcript: tool use, tokens, bench results."""

import json
import re
import sys
from collections import Counter

for path in sys.argv[1:]:
    events = [json.loads(line) for line in open(path) if line.strip()]
    tools = Counter()
    tokens = Counter()
    scores = []
    texts = []
    for event in events:
        part = event.get("part") or {}
        if part.get("type") == "tool":
            tools[part.get("tool")] += 1
            output = json.dumps(part.get("state", {}).get("output", ""))
            scores += re.findall(r"SCORE[^\\\\]*?: ([0-9.]+)", output)
            scores += [f"FAIL:{m}" for m in re.findall(r"\[(\w+)\][^\\\\]*?FAIL", output)]
        if part.get("type") == "step-finish":
            for key in ("input", "output", "reasoning"):
                tokens[key] += part.get("tokens", {}).get(key, 0)
        if part.get("type") == "text" and part.get("text"):
            texts.append(part["text"])
    span = (events[-1]["timestamp"] - events[0]["timestamp"]) / 60000 if events else 0
    print(f"== {path}: {len(events)} events over {span:.1f} min")
    print("   tools:", dict(tools.most_common()))
    print("   tokens:", dict(tokens))
    print("   bench scores seen:", scores[-8:])
    if texts:
        print("   last text:", texts[-1][:400].replace("\n", " "))
