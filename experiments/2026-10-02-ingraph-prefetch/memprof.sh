#!/usr/bin/env bash
# CUDA memory snapshots (rank 0) of startup and of a long-context request: a
# 98K-token uncached prompt (8192-token chunks at growing depth) then 2K more.
#   ./onnode.sh <D>/memprof.sh <tag>   (needs the VLLM_MEM_SNAPSHOT_DIR hook)
cd /e/project1/profound/alint77/vllm
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch; tag="$1"
export VLLM_MEM_SNAPSHOT_DIR=${OUT}/memsnap-${tag} PREFIX_CACHING=1
export SERVE_EXTRA="--profiler-config {\"profiler\":\"cuda\"}"
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server-${tag}.out" 2>"${OUT}/server-${tag}.err" &
pid=$!
ok=
for _ in $(seq 1 360); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { ok=1; break; }
  kill -0 "${pid}" 2>/dev/null || break
  sleep 5
done
[[ -n "${ok}" ]] || { echo "server not ready"; ls "${VLLM_MEM_SNAPSHOT_DIR}"; exit 1; }
echo "ready $(date +%T)"
.venv/bin/python - <<'PY'
import glob, json, time, urllib.request
B = "http://127.0.0.1:8027"
def post(path, body):
    req = urllib.request.Request(B + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800).read()
name = json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"][0]["id"]
text = ""
for f in sorted(glob.glob("vllm/**/*.py", recursive=True)):
    text += open(f, errors="ignore").read()
    if len(text) > 1_000_000:
        break
toks = json.loads(post("/tokenize", {"model": name, "prompt": text}))["tokens"]
post("/start_profile", {})
for n in (98000, 100000):
    t = time.time()
    post("/v1/completions", {"model": name, "prompt": toks[:n], "max_tokens": 8})
    print(f"{n} tokens: {time.time() - t:.1f} s", flush=True)
post("/stop_profile", {})
print("done", flush=True)
PY
kill "${pid}"; sleep 5
for p in $(pgrep -u "${USER}" -f "bin/vllm [s]erve"); do kill "$p" 2>/dev/null; done
wait "${pid}" 2>/dev/null; ls -la "${VLLM_MEM_SNAPSHOT_DIR}"; echo "=== memprof done $(date +%T)"
