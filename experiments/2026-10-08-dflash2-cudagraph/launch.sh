#!/usr/bin/env bash
# One fresh hold per arm, detached:  launch.sh <tag> "<env assignments>"
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-dflash2-cudagraph
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
tag=$1 envs=$2
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
echo "${j} ${tag}"
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "${envs} ${D}/run.sh ${tag}" > ${D}/run-${tag}.log 2>&1
scancel "${j}"
