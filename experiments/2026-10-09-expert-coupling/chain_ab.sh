#!/usr/bin/env bash
# Prod profile (ccf: ccfreq3676) vs steps-touched profile (tch: touch3676-r2000)
# alternating on one fresh hold, prod serve.sh c=1 DFlash2 k=7 via
# ../2026-10-07-mem-reclaim/arm.sh:   chain_ab.sh <first: ccf|tch> <n arms>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-expert-coupling
A=agent_space/experiments/2026-10-07-mem-reclaim/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
arm=$1 n=$2
j=$(sbatch --parsable --time=01:50:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
for i in $(seq 1 "${n}"); do
  prof=glm53-w4a16-agentic-3239-r2000-ccfreq3676.json
  [[ ${arm} == tch ]] && prof=glm53-w4a16-touch3676-r2000.json
  HOLD_JOB=${j} ${E}/onnode.sh "PROFILE=${prof} ${A} ${arm}-${j}-${i}" > ${D}/run-${arm}-${j}-${i}.log 2>&1
  [[ ${arm} == ccf ]] && arm=tch || arm=ccf
done
scancel "${j}"
