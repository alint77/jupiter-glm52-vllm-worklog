#!/usr/bin/env bash
# Alternate eager-draft (cg0) and captured-draft (cg1) prod arms
# (../2026-10-07-mem-reclaim/arm.sh) on one fresh hold:
#   chain_cg.sh <first arm> <n arms>     (KINDS=cg0,cg1 compare_ba.py)
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-dflash2-cudagraph
A=agent_space/experiments/2026-10-07-mem-reclaim/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
first=$1 n=$2
j=$(sbatch --parsable --time=02:00:00 "${E}/hold.sbatch")
echo "${j}"
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
arm=${first}
for i in $(seq 1 "${n}"); do
  e=1; [[ ${arm} == cg1 ]] && e=0
  HOLD_JOB=${j} ${E}/onnode.sh "VLLM_DFLASH2_EAGER_DRAFT=${e} ${A} ${arm}-${j}-${i}" > ${D}/run-${arm}-${j}-${i}.log 2>&1
  [[ ${arm} == cg0 ]] && arm=cg1 || arm=cg0
done
scancel "${j}"
