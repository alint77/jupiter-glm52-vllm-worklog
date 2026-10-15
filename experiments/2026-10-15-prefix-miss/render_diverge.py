"""Server-side prompt tokens of consecutive replayed turns, from a GPU-less
render server (`vllm launch render`, same parsers as serve.sh): converts each
Anthropic body with the server's own converter, posts it to
/v1/chat/completions/render, and prints where consecutive turns' token ids
diverge, with the decoded text around the divergence.

    render_diverge.py 113,114 16,17 63,64 [--port 8091]
"""
import argparse
import json
import urllib.request

from transformers import AutoTokenizer

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages as S

M = "/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-W4A16"
REPLAY = "/e/fscratch/profound/naeimitabiei1/agentic-bench/cc-replay.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(M)
    merge = S._detect_merge_inline_system(open(f"{M}/chat_template.jinja").read())
    want = {int(i) for p in a.pairs for i in p.split(",")}
    rows = {}
    for line in open(REPLAY):
        r = json.loads(line)
        if r["index"] in want:
            rows[r["index"]] = r

    def render(i):
        req = AnthropicMessagesRequest.model_validate(
            dict(rows[i]["body"], model="glm53-w4a16-tiered", stream=False))
        oa = S._convert_anthropic_to_openai_request(req, merge_inline_system=merge)
        body = oa.model_dump(exclude_none=True)
        out = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{a.port}/v1/chat/completions/render",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}), timeout=600).read())
        return out

    cache = {}
    for p in a.pairs:
        ids = [int(x) for x in p.split(",")]
        for i in ids:
            if i not in cache:
                cache[i] = render(i)
                if a.dump:
                    json.dump(cache[i], open(f"{a.dump}/render-{i}.json", "w"))
        for x, y in zip(ids, ids[1:]):
            tx, ty = cache[x]["token_ids"], cache[y]["token_ids"]
            n = 0
            for u, v in zip(tx, ty):
                if u != v:
                    break
                n += 1
            print(f"#{x}->#{y}: {len(tx)} -> {len(ty)} tokens, common prefix {n} "
                  f"({100 * n / len(tx):.1f}% of prev)")
            if n < len(tx):
                print("   prev:", repr(tok.decode(tx[max(0, n - 40):n + 60])))
                print("   next:", repr(tok.decode(ty[max(0, n - 40):n + 60])))


if __name__ == "__main__":
    main()
