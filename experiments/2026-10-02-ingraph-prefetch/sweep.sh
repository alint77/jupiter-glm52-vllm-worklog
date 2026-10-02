#!/usr/bin/env bash
# Build and time several compile-time configurations on one node (same node).
#   ./onnode.sh <D>/sweep.sh "TP_CTAS16=2" "TP_CTAS16=3 TP_DEFER16=0" ...
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-02-ingraph-prefetch
for cfg in "$@"; do
  echo "## $cfg"
  VLLM_TIERED_PREFILL_DEFINES="$cfg" bash $D/run_py.sh $D/bench_m1.py | grep -E "w13 N=( 64| 96|128)"
done
