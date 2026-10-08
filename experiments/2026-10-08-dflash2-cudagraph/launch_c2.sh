#!/usr/bin/env bash
# One fresh hold: ../2026-10-08-c2-k3/run.sh with extra env and probe args.
#   launch_c2.sh <tag> "<env>" [probe args]
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-dflash2-cudagraph
C=agent_space/experiments/2026-10-08-c2-k3
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
X=agent_space/experiments
tag=$1 envs=$2; shift 2
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "${envs} ${C}/run.sh ${tag} --with-profile $*" > ${D}/run-${tag}.log 2>&1
scancel "${j}"
T=/e/fscratch/profound/${USER}/c2-k3/${tag}/trace
for w in ${T}/*/; do
  echo "######## ${w}"
  .venv/bin/python ${X}/2026-09-28-agentic-decode-bench/kernel_tally.py ${w} --top 12
  .venv/bin/python ${X}/2026-10-07-decode-mem-dive/step_tail.py ${w}/*rank0.*.gz | head -8 | cut -c1-200
done > ${D}/analysis-${tag}.txt 2>&1
echo done > ${D}/analysis-${tag}.done
