#!/usr/bin/env bash
# Anchor top-k dump (VLLM_ANCHOR_TOPK_DUMP, topk-dump-pf10.diff on worktree pf10
# = prod 44153148ba): the c=4 agentic timeline (swarm timeline_run.py, 900 s, no
# profiler) at the c=4 launcher config with the touch-order hot set, one fresh
# hold per run (2 runs for variance). Compile caches seeded from mgq-prod.
#   launch_dump.sh             (fresh holds)
#   HOLDS="d1:<job> d2:<job>" launch_dump.sh   (reuse running holds)
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-19-anchor-kv-offload
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
A=agent_space/experiments/2026-10-11-nsys-fresh/arm.sh
S=agent_space/experiments/2026-10-18-c4-swarm-timeline/timeline_run.py
C=/e/fscratch/profound/${USER}/caches
W=/e/fscratch/profound/${USER}/worktrees/pf10
for spec in ${HOLDS:-${RUNS:-d1 d2}}; do (
  run=${spec%%:*}
  if [[ ${spec} == *:* ]]; then j=${spec#*:}; else j=$(sbatch --parsable --time=01:15:00 "${E}/hold.sbatch"); fi
  echo "${run} hold ${j}"
  tag=akv-${run}-${j}
  cb=${C}/akv/vllm-${tag}; tc=${C}/akv/triton-${tag}; mkdir -p ${cb} ${tc}
  rm -rf /e/fscratch/profound/${USER}/sliced-prof/${tag}
  cp -an ${C}/mla-glue-probe/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
  cp -an ${C}/mla-glue-probe/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  HOLD_JOB=${j} ${E}/onnode.sh "${A} ${tag} script PYTHONPATH=${W} PROFILE=glm53-w4a16-sliced-touch-order.json SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} VLLM_ENFORCE_STRICT_TOOL_CALLING=false VLLM_ANCHOR_TOPK_DUMP=/e/fscratch/profound/${USER}/sliced-prof/${tag}/topk SCRIPT=${S} TIMELINE_NOPROF=1 TIMELINE_DURATION=900" > ${X}/logs/${tag}.log 2>&1
  echo "$(date +%T) ${tag}: $(tail -1 ${X}/logs/${tag}.log)"
  echo "=== ${tag} launch done"
) & done
wait
