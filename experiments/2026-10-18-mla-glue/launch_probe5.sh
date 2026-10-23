#!/usr/bin/env bash
# Quality probes of the glue on current prod (worktrees/glue-ab2 =
# 569c42871c: decode_gemm fix + glue, anchors keep their KV write):
# probe.py at the c=4 config, a = glue off, g = VLLM_MLA_QKV_GLUE=1, one
# hold per order, arms in that order, one cache copy per hold.
#   launch_probe5.sh [<order>...]   (default ag ga ag)
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-18-mla-glue
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees
P="PYTHONPATH=${W}/glue-ab2 PROFILE=glm53-w4a16-sliced-touch-order.json"
a_env="${P}"
g_env="${P} VLLM_MLA_QKV_GLUE=1"
orders=("$@"); (( ${#orders[@]} )) || orders=(ag ga ag)
for order in "${orders[@]}"; do (
  j=$(sbatch --parsable --time=$(( ${#order} * 25 + 10 )) "${E}/hold.sbatch")
  echo "${order} hold ${j}"
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  cb=${S}/vllm-hold-${j} tc=${S}/triton-hold-${j}; mkdir -p ${cb} ${tc}
  cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
  cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
  for ((i = 0; i < ${#order}; i++)); do
    arm=${order:i:1} tag=pq5-${order:i:1}-${j}; envv=${arm}_env
    HOLD_JOB=${j} ${E}/onnode.sh "${X}/probe_arm.sh ${tag} 4 4 SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} VLLM_ENFORCE_STRICT_TOOL_CALLING=false ${!envv}" > ${X}/logs/${tag}.log 2>&1
    echo "$(date +%T) ${tag}: $(grep -a 'acc greedy\|=== \|server failed' ${X}/logs/${tag}.log | tail -2 | tr '\n' ' ')"
  done
  rm -rf ${cb} ${tc}
) & done
wait
echo "=== launch_probe5 done"
