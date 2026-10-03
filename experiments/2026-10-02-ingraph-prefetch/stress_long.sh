#!/usr/bin/env bash
# Long-context prefill stress like an agentic session: a ~60K-token prompt,
# then turns growing the same prefix by 2-8K tokens (every 10th turn by 40K,
# a long uncached prefill at depth) up to STRESS_MAX_CTX (default 200K). Survival, TTFT per turn,
# and per-GPU memory sampled every second.
#   ./onnode.sh <D>/stress_long.sh <tag>      (extra env passes to serve.sh)
cd /e/project1/profound/alint77/vllm
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch; tag="$1"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server-${tag}.out" 2>"${OUT}/server-${tag}.err" &
pid=$!
for _ in $(seq 1 360); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 1 >"${OUT}/mem-${tag}.csv" &
smi=$!
.venv/bin/python - "${STRESS_MAX_CTX:-200000}" <<'PY'
import json, random, sys, time, urllib.request
MAX_CTX = int(sys.argv[1])
def post(body):
    req = urllib.request.Request("http://127.0.0.1:8027/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())
name = json.loads(urllib.request.urlopen("http://127.0.0.1:8027/v1/models").read())["data"][0]["id"]
rng = random.Random(0)
ids = [rng.randrange(1000, 100000) for _ in range(60000)]
turn = 0
while len(ids) < MAX_CTX:
    t = time.time()
    try:
        r = post({"model": name, "prompt": ids, "max_tokens": 16, "temperature": 0})
    except Exception as e:
        print(f"turn {turn} at {len(ids)} tokens FAILED: {e}", flush=True); break
    print(f"turn {turn}: {len(ids)} tokens, {time.time() - t:.2f} s, "
          f"cached {r['usage'].get('prompt_tokens_details') or ''}", flush=True)
    n = 40000 if turn % 10 == 9 else rng.choice((2000, 4000, 8000))
    ids += [rng.randrange(1000, 100000) for _ in range(min(n, MAX_CTX - len(ids)))]
    turn += 1
PY
kill "${smi}"
alive=$(kill -0 "${pid}" 2>/dev/null && echo alive || echo dead)
echo "server ${alive}; OOM lines: $(grep -ac 'out of memory' "${OUT}/server-${tag}.err")"
awk -F, '{if ($2 > m[$1]) m[$1] = $2} END {for (g in m) printf "gpu %s peak %d MiB\n", g, m[g]}' "${OUT}/mem-${tag}.csv"
kill "${pid}"; sleep 5
for p in $(pgrep -u "${USER}" -f "bin/vllm [s]erve"); do kill "$p" 2>/dev/null; done
wait "${pid}" 2>/dev/null; echo "=== stress done $(date +%T)"
