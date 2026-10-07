#!/usr/bin/env bash
# Alternate before/after arms on one held node after the arm it is running:
#   chain.sh <job> <before cache index> <wait-for log> <arm>...   (arm: before|after)
# before = HEAD worktree via PYTHONPATH with its own cache root; after = this
# tree with the embedding on Grace at RESERVE_GB=1.7.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-07-mem-reclaim
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
WT=/e/fscratch/profound/${USER}/vllm-before
j=$1; ci=$2; wait_log=$3; shift 3
until grep -qE "=== done|server failed" "${wait_log}" 2>/dev/null; do sleep 20; done
n=2
for arm in "$@"; do
  tag=${arm}-${j}-${n}
  if [[ ${arm} == before ]]; then
    envs="PYTHONPATH=${WT} SERVE_CACHE_ROOT=${C}/vllm-cache-before-${ci}"
  else
    envs="VLLM_TIERED_MOE_EMBED_HOST=1 RESERVE_GB=1.7"
  fi
  HOLD_JOB=${j} ${E}/onnode.sh "${envs} ${D}/arm.sh ${tag}" > ${D}/run-${tag}.log 2>&1
  n=$((n + 1))
done
