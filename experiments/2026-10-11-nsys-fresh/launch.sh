#!/usr/bin/env bash
# Fresh traces of prod (vLLM c8d61506be, frozen worktree nsys2), one fresh
# hold per arm, all in parallel:
#   decode: ../2026-10-11-nsys-sol/launch.sh at c = 1 2 4 8 16
#   prefill sizes and mixed windows at c4: arm.sh
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-11-nsys-fresh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees/nsys2
TREE=nsys2 TAGP=nfr LOGD=${D}/logs agent_space/experiments/2026-10-11-nsys-sol/launch.sh 1 2 4 8 16 &
for what in sizes mixed; do (
  j=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
  cb=${C}/vllm-cache-glm53-nfr-${what}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c16-2264992/{modelinfos,tiered_sliced,torch_extensions} ${cb}/ 2>/dev/null
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  tag=nfr-${what}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/arm.sh ${tag} ${what} PYTHONPATH=${W} SERVE_CACHE_ROOT=${cb}" > ${D}/logs/${tag}.log 2>&1
  scancel "${j}"
) & done
wait
