#!/usr/bin/env bash
# Tiers back to back in prefill (overlap only for <=8 tokens) vs today's
# concurrent default (2048), traced on the same node.
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
D=agent_space/experiments/2026-10-02-ingraph-prefetch
HOLD_JOB=$1 $E/onnode.sh "VLLM_TIERED_MOE_OVERLAP_MAX_TOKENS=8 bash $D/trace.sh seq"
HOLD_JOB=$1 $E/onnode.sh "bash $D/trace.sh conc"
echo "=== seq/conc done $(date +%T)"
