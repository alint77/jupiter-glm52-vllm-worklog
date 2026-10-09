#!/usr/bin/env bash
# Alternate before (c31885ae2d worktree) / after (this tree, c1aec2e22f) c=8
# DFlash2 k=3 sweep arms (1.6M pool) on one fresh hold:
#   chain.sh <first arm: before|after> <n arms> <before cache name>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-dcp-combine32
M=agent_space/experiments/2026-10-08-m32
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
WT=/e/fscratch/profound/${USER}/vllm-before
arm=$1 n=$2 cache=$3
j=$(sbatch --parsable --time=01:40:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
for i in $(seq 1 "${n}"); do
  envs=""
  [[ ${arm} == before ]] && envs="PYTHONPATH=${WT} SERVE_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/marlin/${cache}"
  HOLD_JOB=${j} ${E}/onnode.sh "${envs} ${M}/sweep_arm.sh cmb-${arm}-${j}-${i} dflash2 8 VLLM_TIERED_MOE_KV_POOL_SEQS=4" \
    > ${D}/run-${arm}-${j}-${i}.log 2>&1
  [[ ${arm} == before ]] && arm=after || arm=before
done
scancel "${j}"
