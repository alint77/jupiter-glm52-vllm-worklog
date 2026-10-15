"""How much of an agentic output could prompt-span lookup drafting propose?
(the mlx.fast Bonsai 2 27B track's lookup rounds). For each Claude Code replay
request whose history ends [..., assistant, user], the assistant message is a
past output and everything before it its prompt; both are rendered with the
GLM-5.3 chat template exactly as the fixed server does (vLLM's Anthropic ->
OpenAI conversion, merge_inline_system=False), and the output tokens are the
rendered suffix. Then simulate rounds over the output: if the last L committed
tokens occur exactly once in the prompt, propose the next K prompt tokens
(lookup round: accepted prefix + 1 bonus), else an MTP round worth ACC tokens.

    lookup_cover.py [--L 16] [--K 16] [--acc 2.5] [--limit N]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPLAY = Path("/e/fscratch/profound/naeimitabiei1/agentic-bench/cc-replay.jsonl")
MODEL = "/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-W4A16"


def render(tok, b, msgs, gen):
    return tok.apply_chat_template(msgs, tools=b.get("tools"), add_generation_prompt=gen,
                                   tokenize=True, return_dict=False)


def simulate(ctx, out, L, K, acc):
    index = collections.defaultdict(list)
    for j in range(len(ctx) - L + 1):
        key = tuple(ctx[j:j + L])
        if len(index[key]) < 2:
            index[key].append(j)
    seq = list(ctx)
    t = lk_rounds = lk_tok = mtp_tok = 0
    while t < len(out):
        hits = index.get(tuple(seq[-L:]), []) if len(seq) >= L else []
        if len(hits) == 1 and hits[0] + L < len(ctx):
            prop = ctx[hits[0] + L: hits[0] + L + K]
            a = 0
            while a < len(prop) and t + a < len(out) and prop[a] == out[t + a]:
                a += 1
            n = min(a + 1, len(out) - t)
            if a > 0:
                lk_rounds += 1
                lk_tok += n
                seq += out[t:t + n]
                t += n
                continue
        n = 1  # one MTP round, counted fractionally below
        mtp_tok += n
        seq += out[t:t + n]
        t += n
    return lk_rounds, lk_tok, mtp_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--acc", type=float, default=2.5)
    ap.add_argument("--limit", type=int, default=10**9)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages as S
    tok = AutoTokenizer.from_pretrained(MODEL)
    tot = collections.Counter()
    per = []
    for n, line in enumerate(REPLAY.open()):
        if n >= a.limit:
            break
        r = json.loads(line)
        req = AnthropicMessagesRequest.model_validate(dict(r["body"], model="m", stream=False))
        b = S._convert_anthropic_to_openai_request(req, merge_inline_system=False).model_dump(
            mode="json", exclude_none=True)
        m = b["messages"]
        for x in m:  # as vLLM's renderer: tool-call arguments as dicts
            for tc in x.get("tool_calls") or []:
                if isinstance(tc["function"].get("arguments"), str):
                    tc["function"]["arguments"] = json.loads(tc["function"]["arguments"] or "{}")
        k = max((i for i, x in enumerate(m) if x["role"] == "assistant"), default=None)
        if k is None:
            continue
        ctx = render(tok, b, m[:k], True)
        full = render(tok, b, m[:k + 1], False)
        if full[:len(ctx)] != ctx:
            tot["not_prefix"] += 1
            continue
        out = full[len(ctx):]
        lr, lt, mt = simulate(ctx, out, a.L, a.K, a.acc)
        tot["reqs"] += 1
        tot["out"] += len(out)
        tot["lk_rounds"] += lr
        tot["lk_tok"] += lt
        tot["mtp_tok"] += mt
        per.append((r["index"], len(ctx), len(out), lt))
        print(f"#{r['index']:3d} ctx {len(ctx):6d} out {len(out):5d} lookup-covered {lt:5d} "
              f"({100 * lt / max(1, len(out)):.0f}%) in {lr} rounds", flush=True)
    base = tot["out"] / a.acc
    hyb = tot["lk_rounds"] + tot["mtp_tok"] / a.acc
    print(json.dumps(dict(tot)))
    print(f"L={a.L} K={a.K}: {100 * tot['lk_tok'] / tot['out']:.1f}% of output tokens in lookup "
          f"rounds, {tot['lk_tok'] / max(1, tot['lk_rounds']):.2f} tok/lookup round; rounds "
          f"{base:.0f} -> {hyb:.0f} ({100 * (1 - hyb / base):.1f}% fewer) at MTP {a.acc} tok/round")


if __name__ == "__main__":
    main()
