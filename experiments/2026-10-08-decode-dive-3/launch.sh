#!/usr/bin/env bash
# One fresh hold: decode-mem-dive/run.sh timing (agentic windows + long-context
# windows) on prod serve.sh with extra env, then analyze_all.sh-style analyses.
#   launch.sh <tag> "<env>"
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-decode-dive-3
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
M=agent_space/experiments/2026-10-07-decode-mem-dive
tag=$1 envs=$2
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "${envs} ${M}/run.sh timing ${tag}" > ${D}/run-${tag}.log 2>&1
scancel "${j}"
echo done > ${D}/run-${tag}.done
