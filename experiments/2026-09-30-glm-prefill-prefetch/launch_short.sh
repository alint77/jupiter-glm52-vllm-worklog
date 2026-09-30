#!/usr/bin/env bash
# Short same-node A/B: prefill TTFT sweep (random prompts) + 20 agentic requests
# per arm. Two nodes, arms in opposite orders.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
OUT=/e/fscratch/profound/${USER}/agentic-bench
P=VLLM_TIERED_MOE_COLD_PREFETCH
s1="${P}_SLOTS=1"; s2="${P}_SLOTS=2"; s2x="${P}_SLOTS=2+${P}_MIN_TOKENS=512"
export PREFILL_SWEEP="512 1024 2048 4096" BENCH_ARGS="--limit-requests 20"
n=0
for order in "s1 s2 s2x" "s2x s2 s1"; do
  n=$((n + 1)); j=$(sbatch --parsable --time=01:20:00 "${E}/hold.sbatch"); arms=()
  for a in ${order}; do arms+=("pfs-${a}-n${n}:${!a}"); done
  echo "${j} ${arms[*]}"
  ( bash "${B}/node_ix.sh" "${j}" "${arms[@]}"; scancel "${j}" ) \
    >"${OUT}/node-${j}.out" 2>&1 </dev/null &
done
# Correctness: s2x with every staged layer byte-compared to its Grace source.
j=$(sbatch --parsable --time=00:45:00 "${E}/hold.sbatch"); echo "${j} pfs-vfy"
( BENCH_ARGS="--limit-requests 10" bash "${B}/node_ix.sh" "${j}" \
    "pfs-vfy:${s2x}+${P}_VERIFY=1"; scancel "${j}" ) \
  >"${OUT}/node-${j}.out" 2>&1 </dev/null &
