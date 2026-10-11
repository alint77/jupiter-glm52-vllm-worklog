#!/usr/bin/env bash
# Same-node served A/B of VLLM_SCHED_INTERLEAVE_PREFILL on one fresh hold, one
# arm per argument in order, all from the frozen worktree interleave:
#   ab_launch.sh 0 1 2      (and 2 1 0 on a second node)
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-11-interleave
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees
j=${HOLD:-$(sbatch --parsable --time=02:30:00 "${E}/hold.sbatch")}
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
for k in "$@"; do
  cb=${C}/vllm-cache-glm53-il-k${k}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c16-2264992/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
  tag=il-k${k}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${X}/arm.sh ${tag} ${k} PYTHONPATH=${W}/interleave SERVE_CACHE_ROOT=${cb}" > ${X}/logs/${tag}.log 2>&1
done
scancel "${j}"
