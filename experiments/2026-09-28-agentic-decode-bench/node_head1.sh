#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm
bash agent_space/experiments/2026-09-28-agentic-decode-bench/seq_arms.sh 2113264 "head-1:VLLM_DCP_ONE_SHOT_FUSED=1"
NSYS_OUT=/e/project1/profound/alint77/traces/nsys/glm53-head-2113264 TRACE_ROOT=/e/project1/profound/alint77/traces/nsys/glm53-head-2113264-dummy PROFILE_WINDOWS=1 HOLD_JOB=2113264 agent_space/experiments/2026-09-27-glm53-mtp7-profile/onnode.sh "agent_space/experiments/2026-09-28-agentic-decode-bench/bench_node.sh glm nsys-head --limit-requests 12 --profile-window 3" > agent_space/experiments/2026-09-28-agentic-decode-bench/run-glm-nsys-head.log 2>&1
bash agent_space/experiments/2026-09-28-agentic-decode-bench/seq_arms.sh 2113264 "head-3:VLLM_DCP_ONE_SHOT_FUSED=1"
