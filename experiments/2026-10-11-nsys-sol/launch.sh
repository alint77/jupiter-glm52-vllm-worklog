#!/usr/bin/env bash
# One fresh hold per concurrency, all in parallel, from the frozen Phase 1
# tree (worktrees/fmla-p1 = vllm 80bd3cce5c + the register-split FlashMLA):
#   launch.sh 1 2 4 8 16
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-11-nsys-sol
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
for conc in "$@"; do (
  j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
  cb=${C}/vllm-cache-glm53-nsol-c${conc}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c16-2264992/{modelinfos,tiered_sliced,torch_extensions} ${cb}/ 2>/dev/null
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  tag=nsol-c${conc}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/nsys_arm.sh ${tag} ${conc} PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/fmla-p1 SERVE_CACHE_ROOT=${cb}" > ${D}/logs/${tag}.log 2>&1
  scancel "${j}"
) & done
wait
