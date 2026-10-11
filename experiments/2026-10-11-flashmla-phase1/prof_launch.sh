#!/usr/bin/env bash
# Profiled arms (../2026-10-09-tp-sliced-experts/prof_c8_arm.sh: torch-profiler
# windows at 1x5K, 4x5K, 8x5K, 8x50K; MTP3 tp_sliced c=8) for base (dcp-new)
# and new (fmla-p1), in the given order on one fresh hold.
#   prof_launch.sh base new
cd /e/project1/profound/alint77/vllm
P=agent_space/experiments/2026-10-11-flashmla-phase1
D=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees
declare -A tree=([base]=dcp-new [new]=fmla-p1)
j=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
for arm in "$@"; do
  cb=${C}/vllm-cache-glm53-p1prof-${arm}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c16-2264992/{modelinfos,tiered_sliced,torch_extensions} ${cb}/ 2>/dev/null
  tag=p1prof-${arm}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/prof_c8_arm.sh ${tag} PYTHONPATH=${W}/${tree[${arm}]} SERVE_CACHE_ROOT=${cb} TIERED_MOE_LAYOUT=tp_sliced REPLICAS=" > ${P}/logs/${tag}.log 2>&1
done
scancel "${j}"
