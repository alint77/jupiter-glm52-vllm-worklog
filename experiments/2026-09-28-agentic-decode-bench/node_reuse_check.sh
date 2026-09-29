#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm
until [[ "$(squeue -h -j 2111720 -o %T)" == RUNNING ]]; do sleep 15; done; sleep 10
CUDAGRAPH_MODE=NONE VLLM_DEBUG_CHECK_DCP_INDEX_REUSE=1 PREFIX_CACHING=1 HOLD_JOB=2111720 agent_space/experiments/2026-09-27-glm53-mtp7-profile/onnode.sh "agent_space/experiments/2026-09-28-agentic-decode-bench/bench_node.sh glm reuse-check --limit-requests 8" > agent_space/experiments/2026-09-28-agentic-decode-bench/run-reuse-check.log 2>&1
