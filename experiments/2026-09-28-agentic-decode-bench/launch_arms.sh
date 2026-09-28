#!/usr/bin/env bash
# GLM arms on the MiMo task set, each on its own fresh hold node in parallel,
# prefix caching on (DCP4 unless an arm says otherwise). Detaches; the node is
# released when its arm finishes.
#   bash launch_arms.sh <tag>:<ENV=V+ENV=V> [<tag>:<envs>...]
#   BENCH_ARGS="--limit-requests 40" bash launch_arms.sh ...
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
if [[ "${1:-}" != --detached ]]; then
  jobs_file="${B}/arms-$(date +%H%M%S).txt"
  for arm in "$@"; do
    j=$(sbatch --parsable --time="${HOLD_TIME:-01:15:00}" "${E}/hold.sbatch")
    echo "${j} ${arm%%:*} ${arm#*:}" >>"${jobs_file}"
  done
  cat "${jobs_file}"
  setsid nohup bash "$0" --detached "${jobs_file}" "${BENCH_ARGS:-}" \
    >"${jobs_file%.txt}.out" 2>&1 </dev/null &
  exit 0
fi
jobs_file="$2"; bench_args="$3"
while read -r j tag envs; do
  (
    until [[ "$(squeue -h -j "${j}" -o %T)" == RUNNING ]]; do sleep 20; done
    sleep 10
    env PREFIX_CACHING=1 ${envs//+/ } HOLD_JOB="${j}" "${E}/onnode.sh" \
      "${B}/bench_node.sh glm ${tag} ${bench_args}" >"${B}/run-${tag}.log" 2>&1
    scancel "${j}"
  ) &
done <"${jobs_file}"
wait
