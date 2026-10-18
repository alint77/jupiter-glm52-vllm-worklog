#!/usr/bin/env bash
# Arm b on the "ba" holds, whose first b arms failed on a pf7 missing pf6's
# untracked build outputs: wait for each hold's a arm, then run b.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-18-mla-glue
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/mla-glue
W=/e/fscratch/profound/${USER}/worktrees
b_env="PYTHONPATH=${W}/pf7 PROFILE=glm53-w4a16-sliced-touch-order.json VLLM_MLA_QKV_GLUE=1"
for h in "2392845 tlnsys" "2393363 tlnsys" "2393365 dec1" "2393367 dec4"; do (
  set -- ${h}; j=$1 kind=$2
  until grep -aq "^=== " ${X}/logs/mga-${kind}-${j}.log 2>/dev/null; do sleep 20; done
  tag=mgb-${kind}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} ${kind} ${C}/vllm-${tag} ${C}/triton-${tag} ${b_env}" > ${X}/logs/${tag}.log 2>&1
  echo "$(date +%T) ${tag}: $(grep -a '=== ' ${X}/logs/${tag}.log | tail -1)"
) & done
wait
