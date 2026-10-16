"""Profiled decode and prefill steps at agentic context lengths (c=4 launcher
config), driven by the Claude Code replay (cc-replay.jsonl) converted with
vLLM's own Anthropic -> OpenAI conversion (merge_inline_system=False, as the
fixed server), so prompts render as on /v1/messages.

decode: per context, 4 consecutive replay turns (same session, shared prefix)
        are cached, then streamed together with min_tokens = max_tokens
        (agentic replies are often < 100 tokens); one 2 s Nsight window once
        all 4 are decoding; per-request tok/s outside the window.
prefill: one chunk of N new tokens after a cached replay prompt of length C
        (a user message of repo source appended), max_tokens 1; first timed
        un-profiled on different text, then one Nsight window.

    PROF_WHAT=decode|prefill|decode,prefill prof_longctx.py --out timings.jsonl
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
from conc_probe import call, prompt, spec_counters  # noqa: E402

REPLAY = Path("/e/fscratch/profound/naeimitabiei1/agentic-bench/cc-replay.jsonl")
DECODE = {"80K": (17, 18, 19, 20), "155K": (63, 64, 65, 66),
          "250K": (116, 117, 118, 119), "300K": (161, 163, 164, 165)}
PREFILL = [("fresh", None, 3900), ("80K", 20, 3900), ("155K", 66, 3900),
           ("250K", 119, 3900), ("300K", 165, 3900),
           ("250K", 119, 2000), ("250K", 119, 1000), ("250K", 119, 500)]


def openai_bodies(name, want):
    from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages as S
    out = {}
    for line in REPLAY.open():
        r = json.loads(line)
        if r["index"] in want:
            req = AnthropicMessagesRequest.model_validate(dict(r["body"], model=name, stream=False))
            oa = S._convert_anthropic_to_openai_request(req, merge_inline_system=False)
            b = oa.model_dump(mode="json", exclude_none=True)
            for k in ("max_completion_tokens", "stream_options"):
                b.pop(k, None)
            out[r["index"]] = dict(b, model=name)
    return out


def send(body, **kw):
    t0 = time.time()
    r = json.loads(call("/v1/chat/completions", dict(body, stream=False, **kw)).read())
    return time.time() - t0, r["usage"]["prompt_tokens"]


def stream(body, n, out, started):
    b = dict(body, stream=True, max_tokens=n, min_tokens=n,
             stream_options={"include_usage": True})
    t0 = time.time()
    stamps = []
    usage = None
    with call("/v1/chat/completions", b) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and any(v for k, v in d["choices"][0].get("delta", {}).items()
                                        if k != "role"):
                stamps.append(time.time())
                if len(stamps) == 20:
                    started.release()
    out.update(t0=t0, stamps=stamps, usage=usage)


def decode(name, emit):
    bodies = openai_bodies(name, {i for v in DECODE.values() for i in v})
    for label, idx in DECODE.items():
        for i in idx:
            send(bodies[i], max_tokens=1)
        c0 = spec_counters()
        outs = [dict() for _ in idx]
        started = threading.Semaphore(0)
        th = [threading.Thread(target=stream, args=(bodies[i], 700, o, started))
              for i, o in zip(idx, outs)]
        for t in th:
            t.start()
        for _ in th:
            started.acquire()
        time.sleep(1.0)
        w0 = time.time()
        call("/start_profile").read()
        time.sleep(2.0)
        call("/stop_profile").read()
        w1 = time.time()
        for t in th:
            t.join()
        c1 = spec_counters()
        acc = 1 + (c1["num_accepted_tokens"] - c0["num_accepted_tokens"]) / max(
            1, c1["num_drafts"] - c0["num_drafts"])
        per = []
        for o in outs:
            s = [x for x in o["stamps"] if not (w0 - 0.5 <= x <= w1 + 3)]
            per.append((len(s) - 1) / (s[-1] - s[0]) if len(s) > 1 else None)
        emit({"what": "decode", "label": label, "idx": list(idx),
              "ctx": [o["usage"]["prompt_tokens"] for o in outs],
              "chunks_per_req": [len(o["stamps"]) for o in outs],
              "tokens": [o["usage"]["completion_tokens"] for o in outs],
              "chunk_rate_unprofiled": per,
              "tok_s": [(o["usage"]["completion_tokens"] - 1) / (o["stamps"][-1] - o["stamps"][0])
                        for o in outs], "acc_len": acc})
        time.sleep(3)


def prefill(name, emit):
    bodies = openai_bodies(name, {i for _, i, _ in PREFILL if i is not None})
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    off = 0
    for label, i, n in PREFILL:
        texts = []
        for _ in range(2):
            texts.append(prompt(name, n, corpus[off * 400_000:]))
            off += 1
        if i is not None:
            send(bodies[i], max_tokens=1)
        mk = (lambda t: dict(bodies[i], messages=bodies[i]["messages"] + [
            {"role": "user", "content": t}])) if i is not None else (
            lambda t: {"model": name, "messages": [{"role": "user", "content": t}]})
        s0, p0 = send(mk(texts[0]), max_tokens=1)
        call("/start_profile").read()
        s1, p1 = send(mk(texts[1]), max_tokens=1)
        time.sleep(0.3)
        call("/stop_profile").read()
        time.sleep(3)
        emit({"what": "prefill", "label": f"{label}+{n}", "idx": i,
              "prompt_tokens": [p0, p1], "ttft_unprofiled_s": s0, "ttft_profiled_s": s1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default=os.environ.get("PROF_WHAT", "decode"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    with open(a.out, "w") as f:
        def emit(row):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)
        for what in a.what.split(","):
            {"decode": decode, "prefill": prefill}[what](name, emit)


if __name__ == "__main__":
    main()
