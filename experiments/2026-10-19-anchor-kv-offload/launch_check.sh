#!/usr/bin/env bash
# In-situ correctness of the anchor hot cache at the c=4 launcher config on a
# hold: worktrees/ahc-b1 (e9d65055e3), VLLM_ANCHOR_HOT_CACHE=same with
# VLLM_ANCHOR_HOT_CACHE_CHECK=1 (every mapped anchor FlashMLA call is rerun on
# Grace with the original indices, mismatching elements counted and logged
# every 30 s), dec4 kind (4 decoders at ~5K and ~50K, nsys windows).
#   HOLD_JOB=<j> launch_check.sh [tag-suffix] [extra env...]
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-19-anchor-kv-offload
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/anchor-hot
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees
j=${HOLD_JOB}; sfx=${1:-}; shift || true
tag=ahcchk${sfx}-dec4-${j}
cb=${C}/vllm-${tag} tc=${C}/triton-${tag}; mkdir -p ${cb} ${tc} ${X}/logs
cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
env="PYTHONPATH=${W}/ahc-b1 PROFILE=glm53-w4a16-sliced-touch-order.json VLLM_ANCHOR_HOT_CACHE=same VLLM_ANCHOR_HOT_CACHE_CHECK=1 $*"
HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} dec4 ${cb} ${tc} ${env}" > ${X}/logs/${tag}.log 2>&1
echo "$(date +%T) ${tag}: $(grep -a '=== ' ${X}/logs/${tag}.log | tail -1)"
grep -ah "anchor hot check" /e/fscratch/profound/${USER}/sliced-prof/${tag}/server.* | tail -8
