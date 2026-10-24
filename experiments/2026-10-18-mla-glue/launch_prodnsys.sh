#!/usr/bin/env bash
# nsys of prod with the glue: dec4 (4 requests decoding, M=16 verify steps, at
# ~5K and ~50K) from the main tree at the c=4 launcher config, one cache copy.
#   launch_prodnsys.sh <hold>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-18-mla-glue
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
j=$1
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
cb=${S}/vllm-hold-${j} tc=${S}/triton-hold-${j}; mkdir -p ${cb} ${tc}
cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
tag=prodg-dec4-${j}
echo "$(date +%T) ${tag} start, main tree $(git rev-parse --short HEAD)"
HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} dec4 ${cb} ${tc} PYTHONPATH=$(pwd) PROFILE=glm53-w4a16-sliced-touch-order.json VLLM_MLA_QKV_GLUE=1" > ${X}/logs/${tag}.log 2>&1
echo "$(date +%T) ${tag}: $(grep -a '=== \|server failed' ${X}/logs/${tag}.log | tail -1)"
rm -rf ${cb} ${tc}
echo "=== launch_prodnsys done"
