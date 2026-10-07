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
    TD_TRACE_DIR="${OUT}/td-${name}" TRACE_ROOT="${OUT}/trace-${name}" PROFILE_WINDOWS="${PROFILE_WINDOWS:-1}" \
    "$@" "${E}/onnode.sh" "${B}/bench_node.sh glm ${name} --limit-requests ${n}" \
    >"${OUT}/run-${name}.log" 2>&1 || echo "arm ${name} failed: $?"
}
for a in ${ARMS:-skip base}; do
  case ${a} in
    skip) arm skip SERVE_EXTRA="--mla-cache-tier skip_host_uva" ;;
    stats) arm stats SERVE_EXTRA="--mla-cache-tier skip_host_uva" VLLM_SKIP_KV_STATS=1 ;;
    # the skip tier at about prod's hot count: the copy's cost without the gain
    skipsame) arm skipsame SERVE_EXTRA="--mla-cache-tier skip_host_uva" RESERVE_GB=10.5 ;;
    base) arm base ;;
    dfw) arm dfw DRAFT_QUANT=fp8_per_channel ;;
    # skip-layer KV on Grace + fp8 drafter KV + fp8 drafter weights
    combo) arm combo SERVE_EXTRA="--mla-cache-tier skip_host_uva" DRAFT_KV_DTYPE=fp8 \
      DRAFT_QUANT=fp8_per_channel ;;
    # equal residency (3100 hot per GPU, profile count binding): exactness and
    # the copy's cost, with no hot-set difference between the arms
    eqbase) arm eqbase PROFILE=glm53-w4a16-agentic-3239-r2000-cap3100.json \
      VLLM_TIERED_MOE_PROFILE_CAP=1 ;;
    eqskip) arm eqskip PROFILE=glm53-w4a16-agentic-3239-r2000-cap3100.json \
      VLLM_TIERED_MOE_PROFILE_CAP=1 SERVE_EXTRA="--mla-cache-tier skip_host_uva" ;;
  esac
done
echo "done $(date +%T): ${OUT}"
