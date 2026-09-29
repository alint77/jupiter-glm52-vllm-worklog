#!/usr/bin/env bash
# The GLM ladder re-measured on the MiMo task set: each step is the previous
# one plus one change, from MTP7 as first served to today's baseline. Every arm
# runs the full task set with prefix caching on, REPS times, each on its own
# fresh hold node, all in parallel. Detaches.   bash launch_ladder.sh [REPS]
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
OLD=PROFILE=glm53-w4a16-2496.json
ARMS=(
  "L0-start:DCP=1+REPLICAS=+VLLM_TIERED_MOE_DECODE_KERNEL=0+CAPTURE_SIZES=8+RESERVE_GB=10+${OLD}"
  "L1-capture:DCP=1+REPLICAS=+VLLM_TIERED_MOE_DECODE_KERNEL=0+RESERVE_GB=10+${OLD}"
  "L2-reserve7:DCP=1+REPLICAS=+VLLM_TIERED_MOE_DECODE_KERNEL=0+${OLD}"
  "L3-replicas:DCP=1+VLLM_TIERED_MOE_DECODE_KERNEL=0+${OLD}"
  "L4-onekernel:DCP=1+${OLD}"
  "L5-dcp4:VLLM_DCP_ONE_SHOT_COLLECTIVES=0+${OLD}"
  "L6-oneshot:${OLD}"
  "L7-profile:"
)
if [[ "${1:-}" != --detached ]]; then
  reps="${1:-2}"
  : >"${B}/ladder-jobs.txt"
  for arm in "${ARMS[@]}"; do
    for r in $(seq 1 "${reps}"); do
      j=$(sbatch --parsable --time=01:15:00 "${E}/hold.sbatch")
      echo "${j} ${arm%%:*}-r${r} ${arm#*:}" >>"${B}/ladder-jobs.txt"
    done
  done
  cat "${B}/ladder-jobs.txt"
  setsid nohup bash "$0" --detached >"${B}/launch_ladder.out" 2>&1 </dev/null &
  exit 0
fi
while read -r j tag envs; do
  (
    until [[ "$(squeue -h -j "${j}" -o %T)" == RUNNING ]]; do sleep 20; done
    sleep 10
    env PREFIX_CACHING=1 ${envs//+/ } HOLD_JOB="${j}" "${E}/onnode.sh" \
      "${B}/bench_node.sh glm ${tag}" >"${B}/run-${tag}.log" 2>&1
    scancel "${j}"
  ) &
done <"${B}/ladder-jobs.txt"
wait
