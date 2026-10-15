#!/usr/bin/env bash
# Low-hanging fixes, three fresh holds in parallel:
#   old / new: un-profiled TTFT (ttft_sizes.py) on nsys2 (c8d61506be) vs pf2
#              (7284232eb5, candidate-pack kernel without per-chunk recompile),
#              each with an empty Triton cache, so every new shape costs what
#              it does in prod;
#   pfx:       prefix_repro.py on pf2: consecutive replayed turns at c=1.
# ARMS="new pfx" runs a subset. New worktrees need the build artifacts
# (git ls-files --others --ignored) copied in, or the server fails on import.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-15-c4-timeline
A=agent_space/experiments/2026-10-11-nsys-fresh/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
WT=/e/fscratch/profound/${USER}/worktrees
for arm in ${ARMS:-old new pfx}; do (
  j=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
  cb=${C}/vllm-cache-glm53-fix-${arm}-${j}; mkdir -p ${cb}
  cp -an ${C}/vllm-cache-glm53-c4tl-2333343/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
  tc=/e/fscratch/profound/${USER}/caches/triton-fix-${arm}-${j}; mkdir -p ${tc}
  w=${WT}/pf2; [[ ${arm} == old ]] && w=${WT}/nsys2
  what=ttft; extra="SIZES=512,1000,2000,4000 REPS=4"
  [[ ${arm} == pfx ]] && what=script extra="SCRIPT=${D}/prefix_repro.py"
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  tag=fix-${arm}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${A} ${tag} ${what} PYTHONPATH=${w} SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} ${extra}" > ${D}/logs/${tag}.log 2>&1
  scancel "${j}"
) & done
wait
