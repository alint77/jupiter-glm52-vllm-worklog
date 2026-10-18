#!/usr/bin/env bash
# Long-context arms (ctx_arm.sh / prof_ctx.py) on the holds freed by dec1-4 and
# sizes, same caches:  launch_ctx.sh
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches
arms=("2390686 ctx1 CTX=100,190,380" "2390687 ctx2 CTX=100,190,300"
      "2390688 ctx3 CTX=100,190,250" "2390689 ctx4 CTX=100,190"
      "2390690 ctx1 PREFILL=300")
for a in "${arms[@]}"; do (
  set -- ${a}; j=$1 kind=$2; shift 2
  tag=sw-${kind}$([[ $* == PREFILL* ]] && echo p)-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} ${kind} ${C}/marlin/vllm-cache-glm53-sw-${j} ${C}/triton-sw-${j} $*" > ${D}/logs/${tag}.log 2>&1
  echo "$(date +%T) ${tag}: $(tail -1 ${D}/logs/${tag}.log)"
) & done
wait
