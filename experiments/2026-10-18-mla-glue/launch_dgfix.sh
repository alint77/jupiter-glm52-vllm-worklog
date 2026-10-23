#!/usr/bin/env bash
# Served check of the decode_gemm PDL fix (x loads were hoisted above
# griddepcontrol.wait; race_test.py): probe.py at the c=4 config, arms
#   p = prod (worktrees/pf6)        q = prod + fixed decode_gemm (pf12)
#   g = glue, pf9 knobs off (pf9)   f = glue + fixed decode_gemm (pf13)
# One hold per order, arms in that order on it, one cache copy per hold.
#   launch_dgfix.sh <order>[@<held job>]...   (default pf@2471698 fq qp fp)
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-18-mla-glue
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees
P="PROFILE=glm53-w4a16-sliced-touch-order.json"
p_env="PYTHONPATH=${W}/pf6 ${P}"
q_env="PYTHONPATH=${W}/pf12 ${P}"
g_env="PYTHONPATH=${W}/pf9 ${P} VLLM_MLA_QKV_GLUE=1"
f_env="PYTHONPATH=${W}/pf13 ${P} VLLM_MLA_QKV_GLUE=1"
orders=("$@"); (( ${#orders[@]} )) || orders=(pf@2471698 fq qp fp)
for spec in "${orders[@]}"; do (
  order=${spec%@*} j=${spec#*@}
  [[ ${spec} == *@* ]] || j=$(sbatch --parsable --time=$(( ${#order} * 22 + 10 )) "${E}/hold.sbatch")
  echo "${order} hold ${j}"
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  cb=${S}/vllm-hold-${j} tc=${S}/triton-hold-${j}; mkdir -p ${cb} ${tc}
  cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
  cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
  for ((i = 0; i < ${#order}; i++)); do
    arm=${order:i:1} tag=dgf-${order:i:1}-${j}; envv=${arm}_env
    HOLD_JOB=${j} ${E}/onnode.sh "${X}/probe_arm.sh ${tag} 4 4 SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} VLLM_ENFORCE_STRICT_TOOL_CALLING=false ${!envv}" > ${X}/logs/${tag}.log 2>&1
    echo "$(date +%T) ${tag}: $(grep -a 'acc greedy\|=== \|server failed' ${X}/logs/${tag}.log | tail -2 | tr '\n' ' ')"
  done
  rm -rf ${cb} ${tc}
) & done
wait
echo "=== launch_dgfix done"
