"""Prompt-lookup coverage on GLM's own outputs (thinking, text and tool calls
from timeline_save.py's outputs.jsonl): each output is rendered after its own
replay request's prompt with the GLM chat template; rounds simulated as
../2026-10-16-cudafast-sidequest/lookup_cover.py.

    lookup_glm.py <arm dir> [--L 16] [--K 16] [--acc 3.05]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-16-cudafast-sidequest"))
from lookup_cover import MODEL, REPLAY, render, simulate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--acc", type=float, default=3.05)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages as S
    tok = AutoTokenizer.from_pretrained(MODEL)
    reqs = {json.loads(l)["index"]: json.loads(l) for l in REPLAY.open()}
    ev = {e["index"]: e for e in map(json.loads, (Path(a.arm) / "events.jsonl").open())}
    tot = dict(out=0, usage=0, lk_rounds=0, lk_tok=0, mtp_tok=0, think=0, think_lk=0, n=0)
    for row in map(json.loads, (Path(a.arm) / "outputs.jsonl").open()):
        r, e = reqs[row["index"]], ev.get(row["index"], {})
        if e.get("err"):
            continue
        req = AnthropicMessagesRequest.model_validate(dict(r["body"], model="m", stream=False))
        b = S._convert_anthropic_to_openai_request(req, merge_inline_system=False).model_dump(
            mode="json", exclude_none=True)
        m = b["messages"]
        for x in m:
            for tc in x.get("tool_calls") or []:
                if isinstance(tc["function"].get("arguments"), str):
                    tc["function"]["arguments"] = json.loads(tc["function"]["arguments"] or "{}")
        bl = row["blocks"]
        think = "".join(x["s"] for x in bl if x["type"] == "thinking")
        msg = {"role": "assistant", "content": "".join(x["s"] for x in bl if x["type"] == "text")}
        if think:
            msg["reasoning_content"] = think
        tcs = []
        for x in bl:
            if x["type"] == "tool_use":
                try:
                    args = json.loads(x["s"] or "{}")
                except ValueError:
                    args = {"raw": x["s"]}
                tcs.append({"id": f"c{len(tcs)}", "type": "function",
                            "function": {"name": x["name"], "arguments": args}})
        if tcs:
            msg["tool_calls"] = tcs
        ctx = render(tok, b, m, True)
        full = render(tok, b, m + [msg], False)
        if full[:len(ctx)] != ctx:
            continue
        out = full[len(ctx):]
        lr, lt, mt = simulate(ctx, out, a.L, a.K, a.acc)
        nt = len(tok(think, add_special_tokens=False)["input_ids"]) if think else 0
        tot["n"] += 1
        tot["out"] += len(out)
        tot["usage"] += e.get("output") or 0
        tot["lk_rounds"] += lr
        tot["lk_tok"] += lt
        tot["mtp_tok"] += mt
        tot["think"] += nt
    base = tot["out"] / a.acc
    hyb = tot["lk_rounds"] + tot["mtp_tok"] / a.acc
    print(json.dumps(tot))
    print(f"{Path(a.arm).name} L={a.L} K={a.K}: {tot['n']} outputs, {tot['out']} rendered tokens "
          f"(server counted {tot['usage']}; thinking ~{tot['think']}); "
          f"{100 * tot['lk_tok'] / max(1, tot['out']):.1f}% in lookup rounds, "
          f"{tot['lk_tok'] / max(1, tot['lk_rounds']):.2f} tok/lookup round; rounds {base:.0f} -> "
          f"{hyb:.0f} ({100 * (1 - hyb / base):.1f}% fewer)")


if __name__ == "__main__":
    main()
