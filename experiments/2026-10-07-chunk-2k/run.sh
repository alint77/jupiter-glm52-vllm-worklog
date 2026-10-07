#!/usr/bin/env bash
# One prefill-chunk / reserve arm: prod serve.sh with MAX_NUM_BATCHED_TOKENS and
# RESERVE_GB from the environment; TTFT at agentic and long-context shapes,
# then stress_long.sh's session pattern to 388K. Per-second GPU memory.
#   ./onnode.sh <D>/run.sh <tag>
cd /e/project1/profound/alint77/vllm
OUT=/e/fscratch/profound/${USER}/chunk-2k; mkdir -p "${OUT}"; tag="$1"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server-${tag}.out" 2>"${OUT}/server-${tag}.err" &
pid=$!
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 1 >"${OUT}/mem-${tag}.csv" &
smi=$!
.venv/bin/python - <<'PY'
import json, random, time, urllib.request
def post(body):
    req = urllib.request.Request("http://127.0.0.1:8027/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())
name = json.loads(urllib.request.urlopen("http://127.0.0.1:8027/v1/models").read())["data"][0]["id"]
def toks(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 100000) for _ in range(n)]
def ttft(prompt):
    t = time.time(); r = post({"model": name, "prompt": prompt, "max_tokens": 1, "temperature": 0})
    return time.time() - t, r["usage"]
# warm-up (not timed): a 6K prompt
ttft(toks(6000, 999))
for label, cached, new, reps in (("20K new on 14K cached", 14000, 20000, 3),
                                 ("8K new on 150K cached", 150000, 8000, 2),
                                 ("60K uncached", 0, 60000, 2)):
    times = []
    for rep in range(reps):
        prefix = toks(cached, 1000 + rep) if cached else []
        if cached:
            ttft(prefix)  # cache the prefix
        dt, usage = ttft(prefix + toks(new, 2000 + rep))
        times.append(dt)
        print(f"TTFT {label} rep {rep}: {dt:.3f} s  usage {usage}", flush=True)
    print(f"TTFT {label}: median {sorted(times)[len(times) // 2]:.3f} s  all {[round(t, 3) for t in times]}", flush=True)
# stress_long pattern to 388K
rng = random.Random(0)
ids = [rng.randrange(1000, 100000) for _ in range(60000)]
turn = 0
while len(ids) < 390000:
    t = time.time()
    try:
        r = post({"model": name, "prompt": ids, "max_tokens": 16, "temperature": 0})
    except Exception as e:
        print(f"turn {turn} at {len(ids)} tokens FAILED: {e}", flush=True); break
    print(f"turn {turn}: {len(ids)} tokens, {time.time() - t:.2f} s", flush=True)
    n = 40000 if turn % 10 == 9 else rng.choice((2000, 4000, 8000))
    ids += [rng.randrange(1000, 100000) for _ in range(min(n, 390000 - len(ids)))]
    turn += 1
PY
kill "${smi}"
alive=$(kill -0 "${pid}" 2>/dev/null && echo alive || echo dead)
echo "server ${alive}; OOM lines: $(grep -ac 'out of memory' "${OUT}/server-${tag}.err")"
awk -F, '{if ($2 > m[$1]) m[$1] = $2} END {for (g in m) printf "gpu %s peak %d MiB\n", g, m[g]}' "${OUT}/mem-${tag}.csv"
kill "${pid}"; sleep 5
for p in $(pgrep -u "${USER}" -f "bin/vllm [s]erve"); do kill "$p" 2>/dev/null; done
wait "${pid}" 2>/dev/null; echo "=== run done $(date +%T)"
