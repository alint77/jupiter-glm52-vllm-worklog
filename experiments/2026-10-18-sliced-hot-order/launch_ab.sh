#!/usr/bin/env bash
# ABBA served A/B at the c=4 launcher config: a = prod (worktrees/pf5, ccfreq3676
# profile, round-robin trim), b = worktrees/pf6 (65d9b56fb1, residency order) +
# glm53-w4a16-sliced-touch-order.json. Each hold runs both arms back to back,
# order alternating across holds; fresh compile/triton caches per server
# (AOT caches bake init-time values). Kinds from c4-swarm-timeline/node_run2.sh:
# tlnsys = real c=4 swarm replay with nsys windows, dec<n> = n decoders at
# ~5K / ~50K with nsys windows.   launch_ab.sh
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-18-sliced-hot-order
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/hot-order
W=/e/fscratch/profound/${USER}/worktrees
a_env="PYTHONPATH=${W}/pf5"
b_env="PYTHONPATH=${W}/pf6 PROFILE=glm53-w4a16-sliced-touch-order.json"
holds=("2392844 tlnsys ab" "2392845 tlnsys ba" "2393362 tlnsys ab" "2393363 tlnsys ba"
       "2393364 dec1 ab" "2393365 dec1 ba" "2393366 dec4 ab" "2393367 dec4 ba")
for h in "${holds[@]}"; do (
  set -- ${h}; j=$1 kind=$2 order=$3
  for arm in ${order:0:1} ${order:1:1}; do
    tag=ho${arm}-${kind}-${j}; envv=${arm}_env
    HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} ${kind} ${C}/vllm-${tag} ${C}/triton-${tag} ${!envv}" > ${X}/logs/${tag}.log 2>&1
    echo "$(date +%T) ${tag}: $(grep -a '=== ' ${X}/logs/${tag}.log | tail -1)"
  done
) & done
wait
