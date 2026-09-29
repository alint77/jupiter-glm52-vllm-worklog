#!/usr/bin/env bash
# Run GLM arms one after another on one held node (same-node A/B).
#   seq_arms.sh <job> <tag>:<ENV=V+ENV=V> [...]   (BENCH_ARGS passes through)
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
j="$1"; shift
for arm in "$@"; do
  tag="${arm%%:*}"; envs="${arm#*:}"
  env PREFIX_CACHING=1 ${envs//+/ } HOLD_JOB="${j}" "${E}/onnode.sh" \
    "${B}/bench_node.sh glm ${tag} ${BENCH_ARGS:-}" >"${B}/run-${tag}.log" 2>&1
done
