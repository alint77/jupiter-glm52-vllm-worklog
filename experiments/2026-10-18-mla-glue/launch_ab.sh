#!/usr/bin/env bash
# ABBA served A/B at the c=4 launcher config: a = prod (worktrees/pf6 =
# 65d9b56fb1, touch-order profile), b = worktrees/pf7 (pf6 + glue.diff,
# VLLM_MLA_QKV_GLUE=1). Each hold runs both arms back to back, order
# alternating; fresh compile/triton caches per server. Kinds from
# c4-swarm-timeline/node_run2.sh (tlnsys = c=4 swarm replay, dec<n> = n
# decoders at ~5K / ~50K, nsys windows). Run 2 (mg2*): glue writes the KV only
# in FULL graphs, no splitting op (run 1's split broke piecewise capture).
#   launch_ab.sh
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-18-mla-glue
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/mla-glue2
W=/e/fscratch/profound/${USER}/worktrees
P=PROFILE=glm53-w4a16-sliced-touch-order.json
a_env="PYTHONPATH=${W}/pf6 ${P}"
b_env="PYTHONPATH=${W}/pf7 ${P} VLLM_MLA_QKV_GLUE=1"
holds=("2396112 tlnsys ab" "2396113 tlnsys ba" "2396114 tlnsys ab" "2396115 tlnsys ba"
       "2396116 dec1 ab" "2396117 dec1 ba" "2396118 dec4 ab" "2396119 dec4 ba")
for h in "${holds[@]}"; do (
  set -- ${h}; j=$1 kind=$2 order=$3
  for arm in ${order:0:1} ${order:1:1}; do
    tag=mg2${arm}-${kind}-${j}; envv=${arm}_env
    HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} ${kind} ${C}/vllm-${tag} ${C}/triton-${tag} ${!envv}" > ${X}/logs/${tag}.log 2>&1
    echo "$(date +%T) ${tag}: $(grep -a '=== ' ${X}/logs/${tag}.log | tail -1)"
  done
) & done
wait
