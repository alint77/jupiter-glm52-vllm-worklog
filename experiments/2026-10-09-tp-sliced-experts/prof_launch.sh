#!/usr/bin/env bash
#   prof_launch.sh <ep|sl>: one fresh hold, one profiled arm
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
j=$(sbatch --parsable --time=00:40:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "${D}/prof_arm.sh $1 $1-${j}" > ${D}/logs/serve/prof-$1-${j}.log 2>&1
scancel "${j}"
