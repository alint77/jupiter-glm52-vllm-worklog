#!/usr/bin/env python3
"""Card-protocol acceptance length against an sglang server.

The card defines AL as the per-request mean of completion tokens divided by
verification steps. sglang computes exactly that itself and returns it as
``meta_info["spec_accept_length"]`` (tokenizer_manager.py:2814), so this reads
it per request rather than reconstructing it from counters.

Uses the native /generate endpoint because meta_info is not surfaced on the
OpenAI-compatible route; the chat template is applied client-side so the
prompt is still a real assistant turn.
"""
import argparse, json, statistics, sys, time, urllib.request


def post(base, path, body, timeout=600):
    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", required=True, help="local path, for the tokenizer/chat template")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prompt-field", default="question")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    prompts = []
    with open(a.dataset) as f:
        for line in f:
            if not line.strip():
                continue
            prompts.append(json.loads(line)[a.prompt_field])
            if len(prompts) >= a.num_samples:
                break

    als, rates, verify_cts, comp_tokens = [], [], [], []
    t0 = time.time()
    for i, p in enumerate(prompts):
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        try:
            r = post(a.base, "/generate", {
            "text": text,
            "sampling_params": {"temperature": a.temperature, "top_p": a.top_p,
                                "max_new_tokens": a.max_tokens},
            })
        except Exception as exc:
            # A watchdog-killed or hung server otherwise leaves the client
            # blocked for the whole allocation with nothing to show for it.
            print(f"REQUEST FAILED at sample {i}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 3
        mi = r.get("meta_info", {})
        if "spec_accept_length" in mi:
            als.append(mi["spec_accept_length"])
            rates.append(mi.get("spec_accept_rate", float("nan")))
            verify_cts.append(mi.get("spec_verify_ct", 0))
        comp_tokens.append(mi.get("completion_tokens", 0))
        if i % 16 == 0:
            print(f"  [{i}/{len(prompts)}] al={als[-1] if als else None}", flush=True)
    wall = time.time() - t0

    if not als:
        print("NO spec_accept_length RETURNED -- speculative decoding is off "
              "or the field moved; refusing to emit a number.", file=sys.stderr)
        return 2

    out = {
        "label": a.label,
        "requests": len(prompts),
        "wall_s": round(wall, 2),
        "acceptance_length_mean": round(statistics.mean(als), 4),
        "acceptance_length_median": round(statistics.median(als), 4),
        # Pooled: total completion tokens over total verify steps, the
        # aggregate form. Differs from the per-request mean the card quotes.
        "acceptance_length_pooled": round(sum(comp_tokens) / max(sum(verify_cts), 1), 4),
        "accept_rate_mean": round(statistics.mean(rates), 4),
        "mean_completion_tokens": round(statistics.mean(comp_tokens), 1),
        "total_completion_tokens": sum(comp_tokens),
        "temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens,
        "protocol": "DFlash2 card: chat template, natural EOS, T=1.0 top_p=0.95",
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("\ncard at 7 draft tokens -- GSM8K: DFlash2 5.94 / MTP 5.12")
    return 0


if __name__ == "__main__":
    sys.exit(main())
