#!/usr/bin/env bash
# Anchor KV offload served ABC at the c=4 launcher config, fresh holds:
#   a = prod (worktrees/ahc-a = 24175ed0d3, skip_host_uva)
#   b = worktrees/ahc-b1 + VLLM_ANCHOR_HOT_CACHE=same: anchors' main KV on
#       Grace behind the hot cache, planner charge unchanged (same hot experts)
#   c = the same with VLLM_ANCHOR_HOT_CACHE=free (freed HBM -> hot experts)
#   s = the shipped tree (worktrees/ahc-s1 = dde5b29b6d), VLLM_ANCHOR_HOT_CACHE=1
#   t = worktrees/ahc-s2 (a0ac78d84b: reserved write back on main), the same
# Focus a vs b. Kinds from c4-swarm-timeline/node_run2.sh: tlnsys = real c=4
# swarm replay with nsys windows, dec4 = 4 decoders at ~5K / ~50K. Fresh
# compile/triton caches per server (seeded from prod's).
#   launch_abc.sh ["<kind> <order> [hold]"...]
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-19-anchor-kv-offload
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/anchor-hot
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees
P="PROFILE=glm53-w4a16-sliced-touch-order.json"
a_env="PYTHONPATH=${W}/ahc-a ${P}"
b_env="PYTHONPATH=${W}/ahc-b1 ${P} VLLM_ANCHOR_HOT_CACHE=same"
c_env="PYTHONPATH=${W}/ahc-b1 ${P} VLLM_ANCHOR_HOT_CACHE=free"
s_env="PYTHONPATH=${W}/ahc-s1 ${P} VLLM_ANCHOR_HOT_CACHE=1"
t_env="PYTHONPATH=${W}/ahc-s2 ${P} VLLM_ANCHOR_HOT_CACHE=1"
mkdir -p ${X}/logs
specs=("$@"); (( ${#specs[@]} )) || specs=("tlnsys abc" "tlnsys cba" "dec4 ab" "dec4 ba")
for spec in "${specs[@]}"; do (
  set -- ${spec}; kind=$1 order=$2 j=${3:-}
  [[ -n ${j} ]] || j=$(sbatch --parsable --time=$(( ${#order} * 50 + 15 )) "${E}/hold.sbatch")
  echo "${kind} ${order} hold ${j}"
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  for ((i = 0; i < ${#order}; i++)); do
    arm=${order:i:1} tag=ak${order:i:1}-${kind}-${j}; envv=${arm}_env
    cb=${C}/vllm-${tag} tc=${C}/triton-${tag}; mkdir -p ${cb} ${tc}
    cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
    cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
    HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} ${kind} ${cb} ${tc} ${!envv}" > ${X}/logs/${tag}.log 2>&1
    echo "$(date +%T) ${tag}: $(grep -a '=== ' ${X}/logs/${tag}.log | tail -1)"
  done
) & done
wait
