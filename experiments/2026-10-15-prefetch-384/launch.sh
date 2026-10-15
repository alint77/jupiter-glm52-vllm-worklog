#!/usr/bin/env bash
# Cold-prefetch threshold 384 vs 512 (serve.sh default) at the c4 launcher
# config, vLLM c8d61506be (frozen worktree nsys2), one fresh hold per arm, all
# in parallel: nsys windows (sizes) and un-profiled TTFT (ttft) per threshold.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-15-prefetch-384
A=agent_space/experiments/2026-10-11-nsys-fresh/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees/nsys2
SIZES=320,384,448,512
for thr in 384 512; do for what in sizes ttft; do (
  j=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
  cb=${C}/vllm-cache-glm53-pf${thr}-${what}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c16-2264992/{modelinfos,tiered_sliced,torch_extensions} ${cb}/ 2>/dev/null
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  tag=pf${thr}-${what}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${A} ${tag} ${what} PYTHONPATH=${W} SERVE_CACHE_ROOT=${cb} SIZES=${SIZES} VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=${thr}" > ${D}/logs/${tag}.log 2>&1
  scancel "${j}"
) & done; done
wait
