"""Where do consecutive replayed requests' rendered prompts diverge? Renders
each through vLLM's Anthropic->chat conversion (merge_inline_system as the
server detects it for this template) and the model's chat template, tokenizes,
and prints the common-prefix length with the previous request of the chain.

    prefix_diverge.py IDX [IDX ...]   (pairs (i-1, i) from cc-replay.jsonl)
"""
import json
import sys

from transformers import AutoTokenizer

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages as S

M = "/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-W4A16"
tok = AutoTokenizer.from_pretrained(M)
tmpl = open(f"{M}/chat_template.jinja").read()
merge = S._detect_merge_inline_system(tmpl)
want = set()
for a in sys.argv[1:]:
    want |= {int(a) - 1, int(a)}
rows = {}
for line in open("/e/fscratch/profound/naeimitabiei1/agentic-bench/cc-replay.jsonl"):
    r = json.loads(line)
    if r["index"] in want:
        rows[r["index"]] = r


def render(i):
    body = dict(rows[i]["body"], model="m", stream=False)
    req = AnthropicMessagesRequest.model_validate(body)
    oa = S._convert_anthropic_to_openai_request(req, merge_inline_system=merge)
    msgs = [m if isinstance(m, dict) else m.model_dump(exclude_none=True)
            for m in oa.messages]
    for m in msgs:  # as vLLM's chat path: tool-call arguments as dicts
        if "reasoning" in m and "reasoning_content" not in m:
            m["reasoning_content"] = m.pop("reasoning")
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", tc)
            if isinstance(f.get("arguments"), str):
                f["arguments"] = json.loads(f["arguments"] or "{}")
    tools = [t if isinstance(t, dict) else t.model_dump(exclude_none=True)
             for t in (oa.tools or [])]
    text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False,
                                   add_generation_prompt=True)
    return text, tok(text, add_special_tokens=False)["input_ids"], msgs


print("merge_inline_system:", merge)
for a in sys.argv[1:]:
    i = int(a)
    ta, ia, ma = render(i - 1)
    tb, ib, mb = render(i)
    n = 0
    for x, y in zip(ia, ib):
        if x != y:
            break
        n += 1
    c = 0
    for x, y in zip(ta, tb):
        if x != y:
            break
        c += 1
    print(f"#{i - 1}->#{i}: tokens {len(ia)} -> {len(ib)}, common prefix {n} "
          f"({100 * n / len(ia):.0f}% of prev); roles tail {[m['role'] for m in mb[-4:]]}")
    print("   prev @div:", repr(ta[c - 120:c + 160]))
    print("   next @div:", repr(tb[c - 120:c + 160]))
