#!/usr/bin/env bash
# Long-context SOL: decode (4 live requests at ~80K/155K/250K/300K) and prefill
# (one chunk of 0.5-4K tokens at 0-300K) Nsight windows on pf5 (vLLM
# 144f2f4aee = prod + prefix fix), c=4 launcher config, one fresh hold each.
cd /e/project1/profound/alint77/vllm
P=agent_space/experiments/2026-10-15-longctx-sol
A=agent_space/experiments/2026-10-11-nsys-fresh/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees/pf5
mkdir -p ${P}/logs
for what in ${WHATS:-decode prefill}; do (
  j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
  cb=${C}/vllm-cache-glm53-lc-${what}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c4tl-2333343/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
  tc=/e/fscratch/profound/${USER}/caches/triton-mfix-pfx-2336780
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  tag=lc-${what}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${A} ${tag} script PYTHONPATH=${W} SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} SCRIPT=${P}/prof_longctx.py PROF_WHAT=${what}" > ${P}/logs/${tag}.log 2>&1
  scancel "${j}"
) & done
wait
