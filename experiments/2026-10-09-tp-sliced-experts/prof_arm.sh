#!/usr/bin/env bash
# Profiled c=1 decode arm (prod serve.sh, agentic client flags) from a frozen
# worktree: ep or sl.   ./onnode.sh "<D>/prof_arm.sh <ep|sl> <tag>"
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
C=/e/fscratch/profound/${USER}/caches/marlin
WT=/e/fscratch/profound/${USER}/worktrees
arm=$1 tag=$2
OUT=/e/fscratch/profound/${USER}/sliced-prof/${tag}; mkdir -p "${OUT}/trace"
if [[ ${arm} == ep ]]; then
  export PYTHONPATH=${WT}/ep-base SERVE_CACHE_ROOT=${C}/vllm-cache-glm53-ep-base
else
  export PYTHONPATH=${WT}/tp-sliced-a SERVE_CACHE_ROOT=${C}/vllm-cache-glm53-sliced TIERED_MOE_LAYOUT=tp_sliced REPLICAS=
fi
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export TRACE_ROOT="${OUT}/trace"
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null; kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null' EXIT
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${OUT}"/server.* | sort -u | head; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
.venv/bin/python ${D}/prof_load.py --trace-root "${TRACE_ROOT}"
echo "=== done $(date +%T)"
