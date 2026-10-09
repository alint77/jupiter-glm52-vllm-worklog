#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "SKIP_CHECK=${SKIP_CHECK:-} VARIANTS='${VARIANTS:-}' ${D}/run_bench.sh ${j}" > ${D}/run-${j}.log 2>&1
scancel "${j}"
