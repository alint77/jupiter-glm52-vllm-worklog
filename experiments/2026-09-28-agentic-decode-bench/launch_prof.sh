#!/usr/bin/env bash
# Profile run for one model on a held node, detached from the caller.
#   launch_prof.sh glm|mimo <hold job> [agentic_bench.py args...]
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
model="$1"; job="$2"; shift 2
case "${model}" in
  glm) root=/e/project1/profound/alint77/traces/glm53-agentic-${PROF_TAG:+${PROF_TAG}-}${job} ;;
  mimo) root=/e/project1/profound/alint77/traces/mimo26-agentic-${job} ;;
esac
TRACE_ROOT="${root}" HOLD_JOB="${job}" nohup "${E}/onnode.sh" \
  "${B}/bench_node.sh ${model} ${model}-${PROF_TAG:-prof} $*" >"${B}/run-${model}-${PROF_TAG:-prof}.log" 2>&1 &
echo "${model} profile run on ${job}: traces ${root}, log ${B}/run-${model}-${PROF_TAG:-prof}.log"
