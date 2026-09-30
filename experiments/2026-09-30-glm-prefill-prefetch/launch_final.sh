#!/usr/bin/env bash
# Startup + short bench of the intended defaults: 2 slots, threshold 512,
# capture [8..1024], observed-HBM tolerance 1.5 GB.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
OUT=/e/fscratch/profound/${USER}/agentic-bench
P=VLLM_TIERED_MOE_COLD_PREFETCH
ARM="final-n4:${P}_SLOTS=2+${P}_MIN_TOKENS=512+CAPTURE_SIZES=8,16,32,64,128,256,384,512,640,768,896,1024+VLLM_TIERED_MOE_OBSERVED_HBM_TOLERANCE_GB=1.5"
export PREFILL_SWEEP="512 1024 2048 4096" BENCH_ARGS="--limit-requests 20"
j=$(sbatch --parsable --time=00:45:00 "${E}/hold.sbatch"); echo "node ${j}"
( bash "${B}/node_ix.sh" "${j}" "${ARM}"; scancel "${j}" ) >"${OUT}/node-${j}.out" 2>&1 </dev/null &
