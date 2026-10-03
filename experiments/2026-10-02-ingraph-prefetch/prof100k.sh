#!/usr/bin/env bash
# Prefill + decode at ~100K context: a 98K-token prefix of real code (vLLM
# sources) cached, then ~2K new tokens and 256 generated. Unprofiled reps for
# TTFT / decode rate, then one torch-profiled request (128 tokens out).
#   ./onnode.sh <D>/prof100k.sh <tag>      (extra env passes to serve.sh)
cd /e/project1/profound/alint77/vllm
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch; tag="$1"
export TRACE_ROOT=${OUT}/trace-${tag} PREFIX_CACHING=1
export SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server-${tag}.out" 2>"${OUT}/server-${tag}.err" &
pid=$!
for _ in $(seq 1 360); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
.venv/bin/python - "${PREFIX:-98000}" "${NEW:-2000}" <<'PY'
import glob, json, sys, time, urllib.request
B = "http://127.0.0.1:8027"
def post(path, body, stream=False):
    req = urllib.request.Request(B + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)
name = json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"][0]["id"]
text = ""
for f in sorted(glob.glob("vllm/**/*.py", recursive=True)):
    text += open(f, errors="ignore").read()
    if len(text) > 2_000_000:
        break
toks = json.loads(post("/tokenize", {"model": name, "prompt": text}).read())["tokens"]
P, N = int(sys.argv[1]), int(sys.argv[2])
prefix = toks[:P]
suffix = lambda i: toks[P + i * N: P + (i + 1) * N]
print(f"{len(toks)} tokens of source; prefix {P}, new {N}", flush=True)
t = time.time(); post("/v1/completions", {"model": name, "prompt": prefix, "max_tokens": 1}).read()
print(f"prefix prefill {time.time() - t:.1f} s", flush=True)
def run(i, max_tokens):
    t0 = time.time(); first = None; usage = None
    r = post("/v1/completions", {"model": name, "prompt": prefix + suffix(i),
             "max_tokens": max_tokens, "min_tokens": max_tokens, "stream": True,
             "stream_options": {"include_usage": True}})
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if first is None and d.get("choices") and d["choices"][0].get("text"):
            first = time.time()
        if d.get("usage"):
            usage = d["usage"]
    t1 = time.time()
    n = usage["completion_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"req {i}: prompt {usage['prompt_tokens']} (cached {cached}), TTFT "
          f"{(first - t0) * 1e3:.0f} ms, decode {(n - 1) / (t1 - first):.1f} tok/s "
          f"over {n} tokens", flush=True)
run(0, 256)                    # warm shapes
for i in (1, 2, 3):
    run(i, 256)
post("/start_profile", {}).read()
run(4, 128)
post("/stop_profile", {}).read()
print("traced", flush=True)
PY
kill "${pid}"; sleep 5
for p in $(pgrep -u "${USER}" -f "bin/vllm [s]erve"); do kill "$p" 2>/dev/null; done
wait "${pid}" 2>/dev/null; echo "=== prof100k done $(date +%T)"
