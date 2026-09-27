#!/usr/bin/env bash
# On the held node: serve (serve.sh) with the profiler, capture decode at each
# context (capture.py), stop the server.   ./onnode.sh <E>/run.sh <tag>
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
tag="${1:-$(date +%H%M)}"
export TRACE_ROOT="/e/project1/profound/alint77/traces/glm53-mtp7-${tag}"
mkdir -p "${TRACE_ROOT}"
echo "node $(hostname) traces ${TRACE_ROOT} $(date +%T)"
bash "${E}/serve.sh" >"${E}/server-${tag}.out" 2>"${E}/server-${tag}.err" &
pid=$!
trap 'kill ${pid} 2>/dev/null || true; wait ${pid} 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { grep -ohE "CUDA out of memory.{0,140}|[A-Za-z]*(Error|Exception): .{0,200}" "${E}/server-${tag}".{out,err} | sort -u | head; exit 1; }
  sleep 5
done
echo "server ready $(date +%T)"
grep -ohE "Tiered MoE residency:.{0,60}|GPU KV cache size.{0,30}|observed HBM reserve.{0,50}" "${E}/server-${tag}".{out,err} | sort -u
.venv/bin/python "${E}/capture.py" --trace-root "${TRACE_ROOT}" ${CAPTURE_ARGS:-}
cp "${TRACE_ROOT}/capture-report.json" "${E}/capture-report-${tag}.json"
du -sh "${TRACE_ROOT}"/decode-* | tee "${E}/trace-sizes-${tag}.txt"
echo "=== done $(date +%T) ==="
