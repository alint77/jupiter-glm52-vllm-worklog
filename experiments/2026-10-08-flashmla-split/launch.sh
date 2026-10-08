#!/usr/bin/env bash
# One fresh hold: bench.py at 5K (1280 owned) and 100K (25000 owned) context per rank.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-flashmla-split
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
j=$(sbatch --parsable --time=00:30:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "numactl --cpunodebind=0 --membind=0 .venv/bin/python ${D}/bench.py --owned 1280; numactl --cpunodebind=0 --membind=0 .venv/bin/python ${D}/bench.py --owned 25000" > ${D}/bench-${j}.txt 2>&1
scancel "${j}"
echo done > ${D}/bench.done
