#!/usr/bin/env bash
# Run 3 (mg3*): run 2's b never wrote the KV (the V2 runner captures FULL
# graphs from a NONE-mode forward, so the FULL-mode check never fired; b was
# norms + k RoPE only). c = worktrees/pf8 (pf7 + mla_glue.writes_kv: stream
# capturing with attention metadata). ABBA a vs c on the run-2 holds, each
# hold starting once its run-2 second arm is done.
#   launch_ab3.sh
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-18-mla-glue
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/mla-glue3
W=/e/fscratch/profound/${USER}/worktrees
P=PROFILE=glm53-w4a16-sliced-touch-order.json
a_env="PYTHONPATH=${W}/pf6 ${P}"
c_env="PYTHONPATH=${W}/pf8 ${P} VLLM_MLA_QKV_GLUE=1"
holds=("2396112 tlnsys ac" "2396113 tlnsys ca" "2396114 tlnsys ac" "2396115 tlnsys ca"
       "2396116 dec1 ac" "2396117 dec1 ca" "2396118 dec4 ac" "2396119 dec4 ca")
for h in "${holds[@]}"; do (
  set -- ${h}; j=$1 kind=$2 order=$3
  prev=${X}/logs/mg2$([[ ${order} == ac ]] && echo b || echo a)-${kind}-${j}.log
  until grep -aq "node_run done" ${prev} 2>/dev/null; do sleep 20; done
  sleep 20
  for arm in ${order:0:1} ${order:1:1}; do
    tag=mg3${arm}-${kind}-${j}; envv=${arm}_env
    HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} ${kind} ${C}/vllm-${tag} ${C}/triton-${tag} ${!envv}" > ${X}/logs/${tag}.log 2>&1
    echo "$(date +%T) ${tag}: $(grep -a '=== ' ${X}/logs/${tag}.log | tail -1)"
  done
) & done
wait
