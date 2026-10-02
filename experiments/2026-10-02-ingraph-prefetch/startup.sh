#!/usr/bin/env bash
# Start the GLM server with the given env, report ready/failed, stop it.
#   ./onnode.sh <D>/startup.sh <tag>      (extra env passes through)
cd /e/project1/profound/alint77/vllm
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch; mkdir -p "${OUT}"
tag="$1"
export PREFIX_CACHING="${PREFIX_CACHING-1}"
export SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 ${SERVE_EXTRA:-}"
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server-${tag}.out" 2>"${OUT}/server-${tag}.err" &
pid=$!
for _ in $(seq 1 360); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { echo "READY ${tag} $(date +%T)"; break; }
  kill -0 "${pid}" 2>/dev/null || { echo "FAILED ${tag} $(date +%T)"; break; }
  sleep 5
done
[[ -n "${KEEP:-}" ]] && wait "${pid}"
kill "${pid}" 2>/dev/null; sleep 5
for p in $(pgrep -u "${USER}" -f "bin/vllm [s]erve"); do kill "$p" 2>/dev/null; done
wait "${pid}" 2>/dev/null; true
