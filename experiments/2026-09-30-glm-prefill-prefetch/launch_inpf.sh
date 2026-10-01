#!/usr/bin/env bash
# Prefetch inside prefill graphs (threshold 512, default) vs not (1025), both
# on the serve.sh defaults (2 slots, capture [8..1024]); same-node pairs.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
OUT=/e/fscratch/profound/${USER}/agentic-bench
inpf="VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=512"
nopf="VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=1025"
export PREFILL_SWEEP="512 768 1024 2048" PREFILL_PROMPTS=20 GREEDY_CHECK=1 BENCH_ARGS="--limit-requests 20"
n=0
for order in "inpf nopf" "nopf inpf"; do
  n=$((n + 1)); j=$(sbatch --parsable --time=01:10:00 "${E}/hold.sbatch"); arms=()
  for a in ${order}; do arms+=("gp-${a}-n${n}:${!a}"); done
  echo "${j} ${arms[*]}"
  ( bash "${B}/node_ix.sh" "${j}" "${arms[@]}"; scancel "${j}" ) \
    >"${OUT}/node-${j}.out" 2>&1 </dev/null &
done
