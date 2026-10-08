#!/usr/bin/env bash
# Alternate before (2af85c93c3 worktree) / after (this tree) c=4 arms on one
# fresh hold:  chain_conc.sh <first arm> <n arms>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-m32
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
WT=/e/fscratch/profound/${USER}/vllm-before
first=$1 n=$2
j=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
arm=${first}
for i in $(seq 1 "${n}"); do
  envs=""
  [[ ${arm} == m8 ]] && envs="PYTHONPATH=${WT} SERVE_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/marlin/vllm-cache-before-m32"
  HOLD_JOB=${j} ${E}/onnode.sh "${envs} ${D}/arm_conc.sh ${arm}-${j}-${i}" > ${D}/run-${arm}-${j}-${i}.log 2>&1
  [[ ${arm} == m8 ]] && arm=m32 || arm=m8
done
scancel "${j}"
