#!/usr/bin/env bash
# Capture [8..1024] with the observed-HBM tolerance at 1.5 GB: rerun on hold
# 2125567 (its base arm is done) and a fresh node running both arms.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
OUT=/e/fscratch/profound/${USER}/agentic-bench
CG="CAPTURE_SIZES=8,16,32,64,128,256,384,512,640,768,896,1024+VLLM_TIERED_MOE_OBSERVED_HBM_TOLERANCE_GB=1.5"
export PREFILL_SWEEP="512 1024 2048 4096" BENCH_ARGS="--limit-requests 20"
mkdir -p "${OUT}/stale"; mv -f "${OUT}/run-cg1k-n2.log" "${OUT}/stale/" 2>/dev/null || true
( bash "${B}/seq_arms.sh" 2125567 "cg1k-n2:${CG}"; scancel 2125567 ) >"${OUT}/node-2125567c.out" 2>&1 </dev/null &
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch"); echo "new node ${j}"
( bash "${B}/node_ix.sh" "${j}" "cg1k-n3:${CG}" "cg-base-n3:CAPTURE_SIZES=8"; scancel "${j}" ) \
  >"${OUT}/node-${j}.out" 2>&1 </dev/null &
