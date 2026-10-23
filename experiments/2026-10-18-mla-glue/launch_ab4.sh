#!/usr/bin/env bash
# Run 4 (mg4*): verify-step A/B of the committed glue on current prod, one
# frozen tree (worktrees/glue-ab = d5e946cc43: decode_gemm PDL fix + glue):
#   a = glue off (prod + decode_gemm fix)   g = VLLM_MLA_QKV_GLUE=1
# c=4 launcher config; per hold one kind, both arms back to back, order
# alternating; one cache copy per hold, removed after.
#   [RUN=mg5 TREE=glue-ab2] launch_ab4.sh ["<kind> <order>"...]
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-18-mla-glue
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees
RUN=${RUN:-mg4} TREE=${TREE:-glue-ab}
P="PYTHONPATH=${W}/${TREE} PROFILE=glm53-w4a16-sliced-touch-order.json"
a_env="${P}"
g_env="${P} VLLM_MLA_QKV_GLUE=1"
specs=("$@"); (( ${#specs[@]} )) || specs=("dec1 ag" "dec1 ga" "dec4 ag" "dec4 ga" "tlnsys ag" "tlnsys ga")
for spec in "${specs[@]}"; do (
  set -- ${spec}; kind=$1 order=$2
  j=$(sbatch --parsable --time=70 "${E}/hold.sbatch")
  echo "${kind} ${order} hold ${j}"
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  cb=${S}/vllm-hold-${j} tc=${S}/triton-hold-${j}; mkdir -p ${cb} ${tc}
  cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
  cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
  for arm in ${order:0:1} ${order:1:1}; do
    tag=${RUN}${arm}-${kind}-${j}; envv=${arm}_env
    HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} ${kind} ${cb} ${tc} ${!envv}" > ${X}/logs/${tag}.log 2>&1
    echo "$(date +%T) ${tag}: $(grep -a '=== \|server failed' ${X}/logs/${tag}.log | tail -1)"
  done
  rm -rf ${cb} ${tc}
) & done
wait
echo "=== launch_ab4 ${RUN} done"
