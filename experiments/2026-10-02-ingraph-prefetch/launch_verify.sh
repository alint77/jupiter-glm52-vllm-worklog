#!/usr/bin/env bash
# Byte-level check of the whole-piece design: captured MoEs compare staged vs
# Grace outputs on the device; the eager 2048-token chunk reports them.
cd /e/project1/profound/alint77/vllm
B=agent_space/experiments/2026-09-28-agentic-decode-bench
export BENCH_OUT=/e/fscratch/profound/${USER}/ingraph-prefetch
export PREFILL_SWEEP="512 768 1024 2048" PREFILL_PROMPTS=4 SKIP_AGENTIC=1
until [[ "$(squeue -h -j $1 -o %T)" == RUNNING ]]; do sleep 20; done; sleep 10
bash ${B}/seq_arms.sh "$1" "${TAG:-verify-wide}:VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=512+VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY=1"
echo "=== verify done $(date +%T)"
