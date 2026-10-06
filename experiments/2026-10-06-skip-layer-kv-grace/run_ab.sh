#!/usr/bin/env bash
# skip_host_uva vs prod on one held node, sequential, agentic task set:
#   HOLD_JOB=<id> run_ab.sh <tag> [requests]
# Skip arm first (does it start, what does the planner give, is it exact):
# greedy check, staged-row histogram (VLLM_SKIP_KV_STATS, dumped at each
# profiler start/stop via the local TD_TRACE_DIR hook), one profiler window.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
tag=$1; n=${2:-24}
OUT=/e/fscratch/profound/${USER}/agentic-bench/skipkv-${tag}; mkdir -p "${OUT}"
arm() {  # <name> <extra env...>
  local name=$1; shift
  mkdir -p "${OUT}/td-${name}" "${OUT}/trace-${name}"
  env PREFIX_CACHING=1 GREEDY_CHECK=1 BENCH_OUT="${OUT}" \
    TD_TRACE_DIR="${OUT}/td-${name}" TRACE_ROOT="${OUT}/trace-${name}" PROFILE_WINDOWS=1 \
    "$@" "${E}/onnode.sh" "${B}/bench_node.sh glm ${name} --limit-requests ${n}" \
    >"${OUT}/run-${name}.log" 2>&1 || echo "arm ${name} failed: $?"
}
arm skip SERVE_EXTRA="--mla-cache-tier skip_host_uva" VLLM_SKIP_KV_STATS=1
arm base
echo "done $(date +%T): ${OUT}"
