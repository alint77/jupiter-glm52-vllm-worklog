#!/usr/bin/env bash
# Alternate prod profile (pf0) and the frequency-promoted one (pf1) arms
# (../2026-10-07-mem-reclaim/arm.sh, serve.sh prod defaults) on one held node,
# after <wait-for> logs say "ready":
#   chain_pf.sh <job> "<wait-for log>..." <arm>...   (arm: pf0 | pf1)
cd /e/project1/profound/alint77/vllm
G=agent_space/experiments/2026-10-08-moe-cost-table
A=agent_space/experiments/2026-10-07-mem-reclaim/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
j=$1; waits=$2; shift 2
for w in ${waits}; do
  until grep -qE "^ready|server failed" "${w}" 2>/dev/null; do sleep 20; done
done
n=1
for arm in "$@"; do
  tag=${arm}-${j}-${n}
  prof=glm53-w4a16-agentic-3239-r2000.json
  [[ ${arm} == pf1 ]] && prof=glm53-w4a16-agentic-3239-r2000-ccfreq3676.json
  HOLD_JOB=${j} ${E}/onnode.sh "PROFILE=${prof} ${A} ${tag}" > ${G}/run-${tag}.log 2>&1
  n=$((n + 1))
done
scancel ${j}
