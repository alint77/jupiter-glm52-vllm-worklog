#!/usr/bin/env bash
# One fresh hold per config:  launch_sweep.sh <spec> <c> [extra env...]
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-m32
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
spec=$1 c=$2; shift 2
tag=${spec}3-c${c}${SUFFIX:-}
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "${D}/sweep_arm.sh ${tag} ${spec} ${c} $*" > ${D}/run-sweep-${tag}.log 2>&1
scancel "${j}"
