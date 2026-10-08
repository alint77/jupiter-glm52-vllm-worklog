#!/usr/bin/env bash
# Alternate full-width (dw0) and 768-wide (dw1) DCP sparse decode indices
# (../2026-10-07-mem-reclaim/arm.sh, prod serve.sh) on one fresh hold:
#   chain_dw.sh <first arm> <n arms>     (KINDS=dw0,dw1 compare_ba.py)
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-flashmla-split
A=agent_space/experiments/2026-10-07-mem-reclaim/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
first=$1 n=$2
j=$(sbatch --parsable --time=02:00:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
arm=${first}
for i in $(seq 1 "${n}"); do
  w=0; [[ ${arm} == dw1 ]] && w=768
  HOLD_JOB=${j} ${E}/onnode.sh "VLLM_DCP_SPARSE_DECODE_WIDTH=${w} ${A} ${arm}-${j}-${i}" > ${D}/run-${arm}-${j}-${i}.log 2>&1
  [[ ${arm} == dw0 ]] && arm=dw1 || arm=dw0
done
scancel "${j}"
