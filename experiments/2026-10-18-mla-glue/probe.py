"""Glue correctness probe (bisection on the run-2/3 acceptance drop): fixed
prompts from a frozen corpus snapshot; (1) prompt logprobs of a ~3K prompt
(prefill path), (2) 4 concurrent greedy chats at ~5K, 300 tokens (M=16
decode), (3) 4 concurrent sampled chats as prof_load_n (temperature 1.0,
top_p 0.95, seeds 0-3), each case with the spec-decode counters' acceptance.
Writes <out>/probe.json.

    probe.py --out DIR --corpus FILE
"""
import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
from conc_probe import call, prompt, spec_counters  # noqa: E402

TAIL = "\n\nExplain in detail how the code above is organised."


def chat(name, text, temp, seed, out):
    body = {"model": name, "max_tokens": 300, "ignore_eos": True, "temperature": temp,
            "top_p": 0.95 if temp else 1.0, "seed": seed, "logprobs": True,
            "messages": [{"role": "user", "content":
                          "Below is source code from a project.\n" + text + TAIL}]}
    r = json.loads(call("/v1/chat/completions", body).read())
    ch = r["choices"][0]
    out.update(text=(ch["message"].get("reasoning_content") or "") + "|" + (ch["message"].get("content") or ""),
               lp=[t["logprob"] for t in (ch.get("logprobs") or {}).get("content") or []])


def batch(name, texts, temp):
    c0 = spec_counters()
    outs = [dict() for _ in texts]
    th = [threading.Thread(target=chat, args=(name, t, temp, i, o)) for i, (t, o) in enumerate(zip(texts, outs))]
    [t.start() for t in th]
    [t.join() for t in th]
    c1 = spec_counters()
    d = {k: c1[k] - c0.get(k, 0) for k in c1}
    acc = 1 + d["num_accepted_tokens"] / d["num_drafts"] if d.get("num_drafts") else None
    return {"outs": outs, "acc_len": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", required=True)
    a = ap.parse_args()
    corpus = Path(a.corpus).read_text()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    res = {}
    p3 = prompt(name, 3000, corpus[2_000_000:])
    r = json.loads(call("/v1/completions", {"model": name, "prompt": p3, "max_tokens": 1,
                                             "temperature": 0, "prompt_logprobs": 0}).read())
    res["prompt_lp"] = [None if x is None else list(x.values())[0]["logprob"]
                        for x in r["choices"][0]["prompt_logprobs"]]
    p5 = [prompt(name, 5000, corpus[i * 300_000:]) for i in range(8)]
    for t in p5:
        call("/v1/chat/completions", {"model": name, "max_tokens": 1, "messages": [
            {"role": "user", "content": "Below is source code from a project.\n" + t + TAIL}]}).read()
    g1, g2 = batch(name, p5[:4], 0.0), batch(name, p5[4:], 0.0)
    res["greedy"] = {"outs": g1["outs"] + g2["outs"], "acc_len": (g1["acc_len"] + g2["acc_len"]) / 2}
    res["sampled"] = batch(name, p5[:4], 1.0)
    print("acc greedy %.2f sampled %.2f" % (res["greedy"]["acc_len"], res["sampled"]["acc_len"]), flush=True)
    Path(a.out, "probe.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
