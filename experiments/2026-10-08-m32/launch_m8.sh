#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-m32
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
j=$(sbatch --parsable --time=00:40:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "${D}/grid_m8.sh ${j}" > ${D}/run-m8-${j}.log 2>&1
scancel "${j}"
echo ${j} > ${D}/m8.done
