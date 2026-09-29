#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm
PROF_TAG=head-prof PROFILE_WINDOWS=4 bash agent_space/experiments/2026-09-28-agentic-decode-bench/launch_prof.sh glm 2113265 --limit-requests 60
while pgrep -f "[b]ench_node.sh glm glm-head-prof" >/dev/null; do sleep 20; done
bash agent_space/experiments/2026-09-28-agentic-decode-bench/seq_arms.sh 2113265 "head-2:VLLM_DCP_ONE_SHOT_FUSED=1" "head-4:VLLM_DCP_ONE_SHOT_FUSED=1"
