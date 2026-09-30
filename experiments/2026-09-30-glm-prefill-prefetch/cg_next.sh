#!/usr/bin/env bash
# After the running arm <prev> on hold <job> finishes, run capture [8..1024].
#   cg_next.sh <job> <prev-tag> <new-tag>
cd /e/project1/profound/alint77/vllm
B=agent_space/experiments/2026-09-28-agentic-decode-bench
OUT=/e/fscratch/profound/${USER}/agentic-bench
j=$1; prev=$2; new=$3
until grep -qE "=== done|server failed" "${OUT}/run-${prev}.log" 2>/dev/null; do sleep 20; done
sleep 30
PREFILL_SWEEP="512 1024 2048 4096" BENCH_ARGS="--limit-requests 20" \
  bash "${B}/seq_arms.sh" "${j}" \
  "${new}:CAPTURE_SIZES=8,16,32,64,128,256,384,512,640,768,896,1024"
scancel "${j}"
