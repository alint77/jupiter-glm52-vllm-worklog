#!/usr/bin/env bash
# Prefill CUDA graphs: capture [8] (production) vs [8 .. 2048], same-node pairs,
# prefill TTFT sweep + 20 agentic requests per arm.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
OUT=/e/fscratch/profound/${USER}/agentic-bench
base="CAPTURE_SIZES=8"
cg="CAPTURE_SIZES=8,16,32,64,128,256,512,768,1024,1280,1536,1792,2048"
export PREFILL_SWEEP="512 1024 2048 4096" BENCH_ARGS="--limit-requests 20"
n=0
for order in "base cg" "cg base"; do
  n=$((n + 1)); j=$(sbatch --parsable --time=01:10:00 "${E}/hold.sbatch"); arms=()
  for a in ${order}; do arms+=("cg-${a}-n${n}:${!a}"); done
  echo "${j} ${arms[*]}"
  ( bash "${B}/node_ix.sh" "${j}" "${arms[@]}"; scancel "${j}" ) \
    >"${OUT}/node-${j}.out" 2>&1 </dev/null &
done
