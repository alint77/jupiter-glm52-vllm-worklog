#!/usr/bin/env bash
# Served A/B, prod EP vs tp_sliced, on one fresh hold, arms back to back with
# the standard harness (../2026-10-07-mem-reclaim/arm.sh: GSM8K 200, agentic
# task set, long-context decode, TTFT, 388K stress), each from its own frozen
# worktree:   chain_ab.sh <first arm: ep|sl> <n arms>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
A=agent_space/experiments/2026-10-07-mem-reclaim/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
WT=/e/fscratch/profound/${USER}/worktrees
first=$1 n=$2
j=$(sbatch --parsable --time=02:00:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
arm=${first}
for i in $(seq 1 "${n}"); do
  if [[ ${arm} == ep ]]; then
    envs="PYTHONPATH=${WT}/ep-base SERVE_CACHE_ROOT=${C}/vllm-cache-glm53-ep-base"
  else
    envs="PYTHONPATH=${WT}/tp-sliced-a SERVE_CACHE_ROOT=${C}/vllm-cache-glm53-sliced TIERED_MOE_LAYOUT=tp_sliced REPLICAS="
  fi
  HOLD_JOB=${j} ${E}/onnode.sh "${envs} ${A} ${arm}-${j}-${i}" > ${D}/logs/serve/run-${arm}-${j}-${i}.log 2>&1
  [[ ${arm} == ep ]] && arm=sl || arm=ep
done
scancel "${j}"
