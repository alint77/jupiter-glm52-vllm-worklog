#!/usr/bin/env bash
# One hold: the one-shot collective tests, then a profiled c=8 DFlash2 k=3
# arm on this tree (../2026-10-09-c8-profile/prof_arm.sh) to see the NCCL
# reduce-scatter gone from the 32-token step.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-dcp-combine32
P=agent_space/experiments/2026-10-09-c8-profile
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh ".venv/bin/python -m pytest -q tests/distributed/test_custom_all_reduce.py -k one_shot" > ${D}/run-tests-${j}.log 2>&1
HOLD_JOB=${j} ${E}/onnode.sh "${P}/prof_arm.sh dflash2-k3-combine32 dflash2" > ${D}/run-prof-${j}.log 2>&1
scancel "${j}"
