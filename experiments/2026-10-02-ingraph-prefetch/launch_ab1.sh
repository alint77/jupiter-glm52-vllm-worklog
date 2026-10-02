#!/usr/bin/env bash
# In-graph prefetch (daeecd8: fork at MoE(L), join after it) vs none, same node.
#   launch_ab1.sh <hold job>
cd /e/project1/profound/alint77/vllm
B=agent_space/experiments/2026-09-28-agentic-decode-bench
export BENCH_OUT=/e/fscratch/profound/${USER}/ingraph-prefetch
export PREFILL_SWEEP="512 768 1024" PREFILL_PROMPTS=20 GREEDY_CHECK=1 SKIP_AGENTIC=1
bash ${B}/seq_arms.sh "$1" \
  "ab1-inmoe:VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=512" \
  "ab1-nopf:VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=1025"
echo "=== ab1 done $(date +%T)"
