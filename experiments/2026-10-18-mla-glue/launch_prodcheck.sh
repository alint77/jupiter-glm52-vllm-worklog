#!/usr/bin/env bash
# Prod check of the glue: probe.py from the main tree (what the c=4 launcher
# serves) at the c=4 config with VLLM_MLA_QKV_GLUE=1, on hold <j>, one cache
# copy; then confirm mla_qkv_glue is in the compiled graph.
#   launch_prodcheck.sh <hold>
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-18-mla-glue
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
j=$1
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
cb=${S}/vllm-hold-${j} tc=${S}/triton-hold-${j}; mkdir -p ${cb} ${tc}
cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
tag=prod-g-${j}
echo "$(date +%T) ${tag} start, main tree $(git rev-parse --short HEAD)"
HOLD_JOB=${j} ${E}/onnode.sh "${X}/probe_arm.sh ${tag} 4 4 SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} VLLM_ENFORCE_STRICT_TOOL_CALLING=false PROFILE=glm53-w4a16-sliced-touch-order.json VLLM_MLA_QKV_GLUE=1" > ${X}/logs/${tag}.log 2>&1
echo "$(date +%T) ${tag}: $(grep -a 'acc greedy\|=== \|server failed' ${X}/logs/${tag}.log | tail -2 | tr '\n' ' ')"
echo "glue in compiled graph: $(grep -rl mla_qkv_glue ${cb} 2>/dev/null | wc -l) files"
rm -rf ${cb} ${tc}
echo "=== launch_prodcheck done"
