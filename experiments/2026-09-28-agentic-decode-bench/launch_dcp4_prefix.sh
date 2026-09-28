#!/usr/bin/env bash
# GLM DCP4 400K with prefix caching on (the fp8_ds_mla context-gather fix), the
# full task set on a fresh hold node. Detaches.   bash launch_dcp4_prefix.sh
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
if [[ "${1:-}" != --detached ]]; then
  j=$(sbatch --parsable --time=01:15:00 "${E}/hold.sbatch")
  echo "node ${j}; log ${B}/run-glm-dcp4-prefix.log"
  setsid nohup bash "$0" --detached "${j}" >"${B}/launch_dcp4_prefix.out" 2>&1 </dev/null &
  exit 0
fi
j="$2"
export PREFIX_CACHING=1
until [[ "$(squeue -h -j "${j}" -o %T)" == RUNNING ]]; do sleep 20; done; sleep 10
HOLD_JOB="${j}" "${E}/onnode.sh" "${B}/bench_node.sh glm glm-dcp4-prefix" \
  >"${B}/run-glm-dcp4-prefix.log" 2>&1
