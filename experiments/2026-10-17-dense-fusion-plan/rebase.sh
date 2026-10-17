#!/usr/bin/env bash
# Fusion plan step 1: nsys rebaseline of the dense chain on current prod
# (frozen tree worktrees/pf5 = vllm 144f2f4aee), same arm as 2026-10-11-nsys-sol
# so the per-layer chains compare directly. Runs on an existing hold, never
# cancels it:  rebase.sh <hold id> <concurrency>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-17-dense-fusion-plan
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
j=$1 conc=$2
mkdir -p ${D}/logs
cb=${C}/vllm-cache-glm53-dfp-c${conc}-${j}; mkdir -p ${cb}
cp -an ${C}/vllm-cache-glm53-c4tl-2333343/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
tc=/e/fscratch/profound/${USER}/caches/triton-dfp-c${conc}-${j}; mkdir -p ${tc}
tag=dfp-c${conc}-${j}
HOLD_JOB=${j} ${E}/onnode.sh "agent_space/experiments/2026-10-11-nsys-sol/nsys_arm.sh ${tag} ${conc} PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/pf5 SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc}" > ${D}/logs/${tag}.log 2>&1
echo "$(date +%T) ${tag} finished: $(tail -1 ${D}/logs/${tag}.log)"
