"""Server-side prompt tokens with the Anthropic->OpenAI request passed in
process (as AnthropicServingMessages.create_messages does), not through a
JSON round trip: builds the render server's OnlineRenderer from serve.sh's
parsers and calls render_chat(chat_req) directly. Also renders the JSON
round-tripped request to show which differs.

    MERGE_MODE=server|old_server|template inproc_diverge.py 112,113,114 16,17 63,64

MERGE_MODE picks merge_inline_system: "server" runs the server's own probe
(_should_merge_inline_system, upstream #58754), "old_server" the pre-fix
one (no --chat-template -> merge), "template" GLM's chat_template.jinja.
"""
import asyncio
import json
import os
import sys
from argparse import Namespace

from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages as S
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers import renderer_from_config
from vllm.renderers.online_renderer import OnlineRenderer

M = "/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-W4A16"
REPLAY = "/e/fscratch/profound/naeimitabiei1/agentic-bench/cc-replay.jsonl"


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


async def main(pairs):
    mc = AsyncEngineArgs(model=M, max_model_len=400000,
                         served_model_name=["glm53-w4a16-tiered"]).create_model_config()
    mc.quantization = None
    vc = VllmConfig(model_config=mc)
    r = OnlineRenderer(model_config=mc, renderer=renderer_from_config(vc),
                       request_logger=None, chat_template=None,
                       chat_template_content_format="auto",
                       trust_request_chat_template=False, enable_auto_tools=True,
                       exclude_tools_when_tool_choice_none=False, tool_parser="glm47",
                       reasoning_parser="glm45", default_chat_template_kwargs=None,
                       log_error_stack=False)
    mode = os.environ.get("MERGE_MODE", "template")
    merge = {"old_server": lambda: S._detect_merge_inline_system(None),
             "server": lambda: S._should_merge_inline_system(r),
             "template": lambda: S._detect_merge_inline_system(
                 open(f"{M}/chat_template.jinja").read())}[mode]()
    print("merge mode", mode)
    print("merge_inline_system", merge)
    want = {int(i) for p in pairs for i in p.split(",")}
    rows = {}
    for line in open(REPLAY):
        x = json.loads(line)
        if x["index"] in want:
            rows[x["index"]] = x
    tok = r.renderer.tokenizer if hasattr(r, "renderer") else None

    async def ids_of(req):
        res = await r.render_chat(req)
        _, engine_inputs = res
        return list(engine_inputs[0]["prompt_token_ids"])

    out = {}
    for i in sorted(want):
        req = AnthropicMessagesRequest.model_validate(
            dict(rows[i]["body"], model="glm53-w4a16-tiered", stream=False))
        oa = S._convert_anthropic_to_openai_request(req, merge_inline_system=merge)
        rt = ChatCompletionRequest.model_validate(json.loads(oa.model_dump_json(exclude_none=True)))
        out[i] = (await ids_of(oa), await ids_of(rt))
        print(f"#{i}: in-process {len(out[i][0])}, json round trip {len(out[i][1])}, "
              f"common {common(*out[i])}", flush=True)
    for p in pairs:
        ids = [int(x) for x in p.split(",")]
        for a, b in zip(ids, ids[1:]):
            ta, tb = out[a][0], out[b][0]
            n = common(ta, tb)
            print(f"in-process #{a}->#{b}: {len(ta)} -> {len(tb)}, common {n}")
            if n < len(ta) and tok is not None:
                print("   prev:", repr(tok.decode(ta[max(0, n - 30):n + 50])))
                print("   next:", repr(tok.decode(tb[max(0, n - 30):n + 50])))
    json.dump({k: v[0] for k, v in out.items()}, open(
        f"/e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-15-prefix-miss/renders/inproc-merge{int(merge)}.json", "w"))


asyncio.run(main(sys.argv[1:]))
