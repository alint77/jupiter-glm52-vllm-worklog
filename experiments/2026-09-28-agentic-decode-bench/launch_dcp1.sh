#!/usr/bin/env bash
# GLM DCP1 at MiMo's 250K context, prefix caching on, on the task set: a timing
# run and four profiler windows, each on its own fresh hold node. Detaches, so
# it survives the calling session.   bash launch_dcp1.sh
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
if [[ "${1:-}" != --detached ]]; then
  j1=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
  j2=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
  echo "timing node ${j1}, profile node ${j2}; log ${B}/launch_dcp1.out"
  setsid nohup bash "$0" --detached "${j1}" "${j2}" >"${B}/launch_dcp1.out" 2>&1 </dev/null &
  exit 0
fi
j1="$2"; j2="$3"
export DCP=1 MAX_MODEL_LEN=250000 VLLM_TIERED_MOE_RELAX_SHAPE=1 PREFIX_CACHING=1
until [[ "$(squeue -h -j "${j1}" -o %T)" == RUNNING ]]; do sleep 20; done; sleep 10
HOLD_JOB="${j1}" "${E}/onnode.sh" "${B}/bench_node.sh glm glm-dcp1-250k" \
  >"${B}/run-glm-dcp1-250k.log" 2>&1 &
until [[ "$(squeue -h -j "${j2}" -o %T)" == RUNNING ]]; do sleep 20; done; sleep 10
PROF_TAG=dcp1-prof "${B}/launch_prof.sh" glm "${j2}" --limit-requests 60
wait
