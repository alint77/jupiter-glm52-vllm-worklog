#!/usr/bin/env bash
# M=4 profile (c=4 serving config, 1 request in flight) on a hold once it runs;
# the hold is never cancelled:  m4_on_hold.sh <hold id>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-17-dense-fusion-plan
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
j=$1
until [[ $(squeue -h -j ${j} -o %T 2>/dev/null) == RUNNING ]]; do
  [[ -z $(squeue -h -j ${j} -o %T 2>/dev/null) ]] && { echo "hold ${j} left the queue"; exit 1; }
  sleep 30
done
echo "$(date +%T) hold ${j} RUNNING on $(squeue -h -j ${j} -o %N)"
sleep 5
cb=${C}/vllm-cache-glm53-dfp-c4n1-${j}; mkdir -p ${cb}
cp -an ${C}/vllm-cache-glm53-c4tl-2333343/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
tc=/e/fscratch/profound/${USER}/caches/triton-dfp-c4n1-${j}; mkdir -p ${tc}
tag=dfp-c4n1-${j}
HOLD_JOB=${j} ${E}/onnode.sh "${D}/nsys_arm_n.sh ${tag} 4 1 PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/pf5 SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc}" > ${D}/logs/${tag}.log 2>&1
echo "$(date +%T) ${tag}: $(tail -1 ${D}/logs/${tag}.log)"
