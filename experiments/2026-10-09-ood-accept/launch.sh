#!/usr/bin/env bash
#   launch.sh <tag> <spec> <k>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-ood-accept
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "${D}/ood_arm.sh $1 $2 $3" > ${D}/run-$1.log 2>&1
scancel "${j}"
