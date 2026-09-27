#!/usr/bin/env bash
# Paired GSM8K per arm on a held node: serve.sh with the arm's env, a 2-question
# smoke (a dead server must not score), then N questions with per-question
# outcomes (../2026-09-05-cold-prefetch/gsm8k_paired.py).  ./onnode.sh <E>/gsm.sh tag arm...
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
G=agent_space/experiments/2026-09-05-cold-prefetch/gsm8k_paired.py
tag="$1"; shift
for arm in "$@"; do
  name="${arm%%:*}"; envs="${arm#*:}"
  while curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; do sleep 2; done
  echo "=== ${name} (${envs}) $(date +%T) on $(hostname -s)"
  env ${envs//+/ } bash "${E}/serve.sh" >"${E}/server-gsm-${tag}-${name}.out" 2>"${E}/server-gsm-${tag}-${name}.err" &
  pid=$!
  ok=false
  for _ in $(seq 1 240); do
    curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { ok=true; break; }
    kill -0 "${pid}" 2>/dev/null || break
    sleep 5
  done
  if [[ "${ok}" == true ]]; then
    .venv/bin/python "${G}" --num-questions 2 --out "${E}/gsm-${tag}-${name}-smoke.json" &&
      .venv/bin/python "${G}" --num-questions "${NQ:-1000}" --out "${E}/gsm-${tag}-${name}.json" || true
  else
    echo "server failed"
  fi
  kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
done
echo "=== done $(date +%T)"
