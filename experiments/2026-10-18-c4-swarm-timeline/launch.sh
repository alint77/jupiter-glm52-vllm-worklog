#!/usr/bin/env bash
# The campaign: one arm per hold, all at once, own compile + Triton caches
# (seeded from the pf5 c=4 cache). Holds are never cancelled.
#   launch.sh            (holds 2390683-2390691)
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches
WA="30:6,120:6,240:6,360:6,480:6,600:6,720:6"
WB="60:6,180:6,300:6,420:6,540:6,660:6,780:6"
arms=(
  "2390683 tlpyspy TIMELINE_DURATION=900"
  "2390684 tlmem TIMELINE_DURATION=900"
  "2390685 tlnsys TIMELINE_DURATION=900 TIMELINE_WINDOWS=${WA}"
  "2390691 tlnsys TIMELINE_DURATION=900 TIMELINE_WINDOWS=${WB}"
  "2390686 dec1"
  "2390687 dec2"
  "2390688 dec3"
  "2390689 dec4"
  "2390690 sizes SIZES=512,1000,2000,4000"
)
for a in "${arms[@]}"; do (
  set -- ${a}; j=$1 kind=$2; shift 2
  tag=sw-${kind}-${j}
  cb=${C}/marlin/vllm-cache-glm53-sw-${j}; mkdir -p ${cb}
  cp -an ${C}/marlin/vllm-cache-glm53-dfp-c4n1-2383231/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
  tc=${C}/triton-sw-${j}; mkdir -p ${tc}
  cp -an ${C}/triton-dfp-c4n1-2383231/. ${tc}/ 2>/dev/null
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run.sh ${tag} ${kind} ${cb} ${tc} $*" > ${D}/logs/${tag}.log 2>&1
  echo "$(date +%T) ${tag}: $(tail -1 ${D}/logs/${tag}.log)"
) & done
wait
