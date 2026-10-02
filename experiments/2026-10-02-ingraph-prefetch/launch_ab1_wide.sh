#!/usr/bin/env bash
# Third arm on the same node as launch_ab1.sh: fork before o_proj(L), join
# after the projections of L+1, first MoE layer staged under the dense layer.
cd /e/project1/profound/alint77/vllm
B=agent_space/experiments/2026-09-28-agentic-decode-bench
export BENCH_OUT=/e/fscratch/profound/${USER}/ingraph-prefetch
export PREFILL_SWEEP="512 768 1024" PREFILL_PROMPTS=20 GREEDY_CHECK=1 SKIP_AGENTIC=1
until grep -q "ab1 done" "${BENCH_OUT}/ab1.log"; do sleep 10; done
bash ${B}/seq_arms.sh "$1" "ab1-wide:VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=512"
echo "=== wide done $(date +%T)"
