#!/usr/bin/env bash
# Run one GLM arm on an already-submitted hold job once it starts, then release
# it. Detached by the caller.   use_hold.sh <job> <tag> <ENV=V+ENV=V> [bench args]
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
j="$1"; tag="$2"; envs="$3"; shift 3
until [[ "$(squeue -h -j "${j}" -o %T)" == RUNNING ]]; do
  [[ -z "$(squeue -h -j "${j}" -o %T)" ]] && exit 1
  sleep 20
done
sleep 10
env PREFIX_CACHING=1 ${envs//+/ } HOLD_JOB="${j}" "${E}/onnode.sh" \
  "${B}/bench_node.sh glm ${tag} $*" >"${B}/run-${tag}.log" 2>&1
scancel "${j}"
