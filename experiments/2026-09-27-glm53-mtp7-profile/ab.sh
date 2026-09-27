#!/usr/bin/env bash
# Same-node A/B on a held node: for each arm (NAME:ENV...), start serve.sh with
# that env, run bench.py (greedy, fixed prompts), stop.  ./onnode.sh <E>/ab.sh tag arm...
#   arm example:  cg18:CAPTURE_SIZES=1,8   r7:CAPTURE_SIZES=1,8+RESERVE_GB=7
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
tag="$1"; shift
for arm in "$@"; do
  name="${arm%%:*}"; envs="${arm#*:}"
  # the previous arm's server must be gone before /health can be trusted
  while curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; do sleep 2; done
  echo "=== ${name} (${envs}) $(date +%T) on $(hostname -s)"
  env ${envs//+/ } \
    bash "${E}/serve.sh" >"${E}/server-ab-${tag}-${name}.out" 2>"${E}/server-ab-${tag}-${name}.err" &
  pid=$!
  ok=false
  for _ in $(seq 1 240); do
    curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { ok=true; break; }
    kill -0 "${pid}" 2>/dev/null || break
    sleep 5
  done
  if [[ "${ok}" == true ]]; then
    grep -m1 -ohE "Tiered MoE residency: [0-9]+ hot" "${E}/server-ab-${tag}-${name}.err" || true
    .venv/bin/python "${E}/bench.py" --out "${E}/ab-${tag}-${name}.json" ${BENCH_ARGS:-} || true
  else
    echo "server failed"; grep -ohE "[A-Za-z]*Error: .{0,160}" "${E}/server-ab-${tag}-${name}.err" | sort -u | head -3
  fi
  kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
done
echo "=== done $(date +%T)"
