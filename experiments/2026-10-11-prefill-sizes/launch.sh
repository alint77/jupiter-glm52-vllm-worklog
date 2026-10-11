#!/usr/bin/env bash
# One fresh hold per "budget:sizes" argument, in parallel, from the frozen
# worktree interleave:   launch.sh 4096:500,1000,2000,4000 8192:500,4000,8000
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-11-prefill-sizes
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees
for spec in "$@"; do (
  budget=${spec%%:*} sizes=${spec#*:}
  j=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
  cb=${C}/vllm-cache-glm53-pfs-${budget}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c16-2264992/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  tag=pfs-b${budget}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/arm.sh ${tag} ${budget} ${sizes} PYTHONPATH=${W}/interleave SERVE_CACHE_ROOT=${cb}" > ${D}/logs/${tag}.log 2>&1
  scancel "${j}"
) & done
wait
