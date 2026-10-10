#!/usr/bin/env bash
# The prod GLM-5.3 server (../2026-09-27-glm53-mtp7-profile/serve.sh) with the
# routed experts TP-sliced: --tiered-moe-layout tp_sliced (EP off), no replicas.
# Imports vLLM from a frozen worktree:  WT=<worktree> serve-sliced.sh
set -euo pipefail
export PYTHONPATH="${WT:?frozen worktree}"
export TIERED_MOE_LAYOUT=tp_sliced
export REPLICAS=
export SERVE_CACHE_ROOT="${SERVE_CACHE_ROOT:-/e/fscratch/profound/${USER}/caches/marlin/vllm-cache-glm53-sliced}"
exec /e/project1/profound/alint77/vllm/agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh
