#!/usr/bin/env bash
# Both phases on one existing hold, one server start:
#   run_hold.sh <hold job id>
cd /e/project1/profound/alint77/vllm
P=agent_space/experiments/2026-10-15-longctx-sol
A=agent_space/experiments/2026-10-11-nsys-fresh/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees/pf5
j=$1
mkdir -p ${P}/logs
cb=${C}/vllm-cache-glm53-lc-${j}; mkdir -p ${cb}
cp -an ${C}/vllm-cache-glm53-c4tl-2333343/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
tc=/e/fscratch/profound/${USER}/caches/triton-mfix-pfx-2336780
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
tag=lc-${j}
HOLD_JOB=${j} ${E}/onnode.sh "${A} ${tag} script PYTHONPATH=${W} SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} SCRIPT=${P}/prof_longctx.py PROF_WHAT=decode,prefill" > ${P}/logs/${tag}.log 2>&1
echo "hold ${j} left allocated: $(squeue -h -j ${j} -o %L) left"
