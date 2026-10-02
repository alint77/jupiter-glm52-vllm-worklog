#!/usr/bin/env bash
# Prefill MoE kernel (VLLM_TIERED_MOE_PREFILL_KERNEL=1) vs Marlin, same node:
# TTFT sweep, greedy outputs, GSM8K 400. Arm order alternates per pass.
#   launch_pk.sh <hold job> <pass>
cd /e/project1/profound/alint77/vllm
B=agent_space/experiments/2026-09-28-agentic-decode-bench
export BENCH_OUT=/e/fscratch/profound/${USER}/ingraph-prefetch
export PREFILL_SWEEP="512 768 1024 2048 4096" PREFILL_PROMPTS=20 GREEDY_CHECK=1 SKIP_AGENTIC=1
export GSM8K_N=${GSM8K_N-400}
# one compile cache per hold: servers on several nodes writing one cache
# corrupt each other's torch.compile artifacts (2026-10-02)
c=/e/fscratch/profound/${USER}/caches/marlin
export SERVE_CACHE_ROOT=${c}/vllm-cache-glm53-mtp7-$1
mkdir -p ${SERVE_CACHE_ROOT}; cp -rn ${c}/vllm-cache-glm53-mtp7/torch_extensions ${SERVE_CACHE_ROOT}/
a="pk$2-new:VLLM_TIERED_MOE_PREFILL_KERNEL=1"; b="pk$2-old:VLLM_TIERED_MOE_PREFILL_KERNEL=0"
if (( $2 % 2 )); then bash ${B}/seq_arms.sh "$1" "$a" "$b"; else bash ${B}/seq_arms.sh "$1" "$b" "$a"; fi
echo "=== pk$2 done $(date +%T)"
