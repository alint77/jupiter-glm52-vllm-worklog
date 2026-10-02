#!/usr/bin/env bash
# Torch-profile one prefill each at TRACE_SIZES new tokens (default 512 768 1024).
#   ./onnode.sh <D>/trace.sh <tag>   -> traces in fscratch/ingraph-prefetch/trace-<tag>
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
.venv/bin/python - ${TRACE_SIZES:-512 768 1024} <<'PY'
import json, random, sys, urllib.request
SIZES = [int(a) for a in sys.argv[1:]]
def post(path, body=None):
    req = urllib.request.Request("http://127.0.0.1:8027" + path,
        data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600).read()
name = json.loads(urllib.request.urlopen("http://127.0.0.1:8027/v1/models").read())["data"][0]["id"]
def prefill(n, seed):
    rng = random.Random(seed)
    ids = [rng.randrange(1000, 100000) for _ in range(n)]
    post("/v1/completions", {"model": name, "prompt": ids, "max_tokens": 1})
for n in SIZES:
    prefill(n, n)          # warm, different prompt from the traced one
post("/start_profile")
for n in SIZES:
    prefill(n, n + 1)
post("/stop_profile")
print("traced")
PY
kill "${pid}"; sleep 5
for p in $(pgrep -u "${USER}" -f "bin/vllm [s]erve"); do kill "$p" 2>/dev/null; done
wait "${pid}" 2>/dev/null; echo "=== trace done $(date +%T)"
