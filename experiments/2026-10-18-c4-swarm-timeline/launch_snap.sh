#!/usr/bin/env bash
# The allocator-snapshot arm (tlsnap) on its own hold:  launch_snap.sh <hold>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches
j=$1 tag=sw-tlsnap-$1
cb=${C}/marlin/vllm-cache-glm53-sw-${j}; mkdir -p ${cb}
cp -an ${C}/marlin/vllm-cache-glm53-dfp-c4n1-2383231/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
tc=${C}/triton-sw-${j}; mkdir -p ${tc}
cp -an ${C}/triton-dfp-c4n1-2383231/. ${tc}/ 2>/dev/null
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run.sh ${tag} tlsnap ${cb} ${tc} TIMELINE_DURATION=900 TIMELINE_WINDOWS=300:1,600:1,880:1" > ${D}/logs/${tag}.log 2>&1
echo "$(date +%T) ${tag}: $(tail -1 ${D}/logs/${tag}.log)"
