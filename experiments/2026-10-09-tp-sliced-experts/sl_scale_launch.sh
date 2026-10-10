#!/usr/bin/env bash
# tp_sliced conc_scale arm at 64 in flight (scale_arm2.sh), one fresh hold:
#   sl_scale_launch.sh <reserve GB>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
res=$1
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
cb=${C}/vllm-cache-glm53-sc-sl-${j}; mkdir -p ${cb}; cp -an ${C}/vllm-cache-glm53-sliced/{modelinfos,torch_extensions} ${cb}/ 2>/dev/null
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
tag=scale-mtp3-r${res}-sl64-${j}
HOLD_JOB=${j} ${E}/onnode.sh "${D}/scale_arm2.sh ${tag} SEQS=64 RESERVE_GB=${res} PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/tp-sliced-scale SERVE_CACHE_ROOT=${cb} TIERED_MOE_LAYOUT=tp_sliced REPLICAS=" > ${D}/logs/serve/${tag}.log 2>&1
scancel "${j}"
