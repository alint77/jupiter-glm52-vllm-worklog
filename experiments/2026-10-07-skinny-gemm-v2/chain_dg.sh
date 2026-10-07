#!/usr/bin/env bash
# Alternate VLLM_DECODE_GEMM=0 / 1 arms (../2026-10-07-mem-reclaim/arm.sh,
# serve.sh prod defaults) on one held node, after <wait-for> logs say "ready":
#   chain_dg.sh <job> "<wait-for log>..." <arm>...   (arm: dg0 | dg1)
cd /e/project1/profound/alint77/vllm
P=agent_space/experiments/2026-10-07-skinny-gemm-v2
A=agent_space/experiments/2026-10-07-mem-reclaim/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
j=$1; waits=$2; shift 2
for w in ${waits}; do
  until grep -qE "^ready|server failed" "${w}" 2>/dev/null; do sleep 20; done
done
n=1
for arm in "$@"; do
  tag=${arm}-${j}-${n}
  HOLD_JOB=${j} ${E}/onnode.sh "VLLM_DECODE_GEMM=${arm#dg} ${A} ${tag}" > ${P}/run-${tag}.log 2>&1
  n=$((n + 1))
done
