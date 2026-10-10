#!/usr/bin/env bash
# c16_arm.sh on one fresh hold from the frozen worktree tp-sliced-c16:
#   c16_launch.sh <reserve GB> [env...]
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
res=$1; shift
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
cb=${C}/vllm-cache-glm53-c16-${j}; mkdir -p ${cb}; cp -an ${C}/vllm-cache-glm53-sliced/{modelinfos,torch_extensions} ${cb}/ 2>/dev/null
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
tag=c16-mtp3-r${res}-${j}
HOLD_JOB=${j} ${E}/onnode.sh "${D}/c16_arm.sh ${tag} RESERVE_GB=${res} PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/tp-sliced-c16 SERVE_CACHE_ROOT=${cb} $*" > ${D}/logs/serve/${tag}.log 2>&1
scancel "${j}"
