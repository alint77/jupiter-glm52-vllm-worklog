#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm
while pgrep -f "[n]ode_a.sh" >/dev/null; do sleep 20; done
sleep 20
CUDAGRAPH_MODE=NONE TRACE_STACK=true PROF_TAG=eager-stack PROFILE_WINDOWS=1 TRACE_ROOT=/e/project1/profound/alint77/traces/glm53-agentic-eager-stack-2111453 HOLD_JOB=2111453 agent_space/experiments/2026-09-27-glm53-mtp7-profile/onnode.sh "agent_space/experiments/2026-09-28-agentic-decode-bench/bench_node.sh glm glm-eager-stack --limit-requests 12 --profile-window 5 --profile-min-steps 8" > agent_space/experiments/2026-09-28-agentic-decode-bench/run-glm-eager-stack.log 2>&1
