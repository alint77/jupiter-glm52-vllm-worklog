#!/usr/bin/env bash
# Routing-record arms (worktrees/pf5rt = pf5 + route_record.py: per-step routed
# expert ids of every sliced decode MoE call, dumped at each profiler start/stop
# to <OUT>/route) on the free holds, same caches:  launch_rt.sh
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches
arms=("2390685 ctx4 CTX=100,190" "2390683 tlnsys" "2390684 dec4" "2390691 dec2")
for a in "${arms[@]}"; do (
  set -- ${a}; j=$1 kind=$2; shift 2
  tag=rt-${kind}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} ${kind} ${C}/marlin/vllm-cache-glm53-sw-${j} ${C}/triton-sw-${j} $* PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/pf5rt VLLM_ROUTE_RECORD=512 VLLM_ROUTE_RECORD_DIR=/e/fscratch/profound/${USER}/sliced-prof/${tag}/route" > ${D}/logs/${tag}.log 2>&1
  echo "$(date +%T) ${tag}: $(tail -1 ${D}/logs/${tag}.log)"
) & done
wait
